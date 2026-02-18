import argparse
from pathlib import Path
from vllm import LLM, SamplingParams
from pydantic import BaseModel, Field
from fastapi import FastAPI
import uvicorn


# ── Request / Response models ──────────────────────────────────────────

class RolloutRequest(BaseModel):
    messages: list[list[dict[str, str]]] = Field(..., description="Batch of conversations (list of message lists)")
    max_tokens: int = Field(256, description="Max tokens to generate per completion")
    temperature: float = Field(0.7, description="Sampling temperature")
    top_p: float = Field(1.0, description="Top-p (nucleus) sampling")
    top_k: int = Field(-1, description="Top-k sampling (-1 = disabled)")


class RolloutResponse(BaseModel):
    prompt: list[str]
    completion: list[str]
    completion_token_ids: list[list[int]]
    old_log_probs: list[list[float]]


# ── vLLM Client ────────────────────────────────────────────────────────

class VLLMClient:
    def __init__(self, model_path: str, tensor_parallel_size: int = 1):
        self.model_path = model_path
        self.llm = LLM(
            model=model_path,
            tensor_parallel_size=tensor_parallel_size,
        )
        self.tokenizer = self.llm.get_tokenizer()

    def _get_sampling_params(self, request: RolloutRequest) -> SamplingParams:
        return SamplingParams(
            temperature=request.temperature,
            max_tokens=request.max_tokens,
            top_p=request.top_p,
            top_k=request.top_k,
            logprobs=0,          # only the sampled token's logprob
            prompt_logprobs=0,   # only the prompt token's logprob
        )

    def _build_rollout_response(self, outputs: list) -> RolloutResponse:
        prompts = []
        completions = []
        all_token_ids = []
        all_log_probs = []

        for output in outputs:
            completion_text = output.outputs[0].text
            completion_text+=self.tokenizer.eos_token
            prompts.append(output.prompt)
            completions.append(completion_text)

            # Use vLLM's own token IDs (guaranteed to match logprobs 1:1)
            token_ids = list(output.outputs[0].token_ids)
            all_token_ids.append(token_ids)

            # Extract per-token log probs for the sampled token
            log_probs = []
            for token_logprob_dict in output.outputs[0].logprobs:
                sampled_logprob = next(iter(token_logprob_dict.values())).logprob
                log_probs.append(sampled_logprob)
            all_log_probs.append(log_probs)

            assert len(token_ids) == len(log_probs), (
                f"Token count mismatch: {len(token_ids)} token_ids vs {len(log_probs)} logprobs"
            )

        return RolloutResponse(
            prompt=prompts,
            completion=completions,
            completion_token_ids=all_token_ids,
            old_log_probs=all_log_probs,
        )

    def generate(self, request: RolloutRequest) -> RolloutResponse:
        sampling_params = self._get_sampling_params(request)
        prompts = [
            self.tokenizer.apply_chat_template(conv, tokenize=False, add_generation_prompt=True)
            for conv in request.messages
        ]
        outputs = self.llm.generate(prompts, sampling_params)
        return self._build_rollout_response(outputs)

    def reload_weights(self):
        self.llm.collective_rpc("reload_weights")


# ── FastAPI server ─────────────────────────────────────────────────────

app = FastAPI(title="vLLM Policy Server")
client: VLLMClient | None = None


@app.post("/generate", response_model=RolloutResponse)
def generate(request: RolloutRequest):
    return client.generate(request)


@app.post("/reload_weights")
def reload_weights():
    client.reload_weights()
    return {"status": "ok", "message": "Weights reloaded"}


@app.get("/health")
def health():
    return {"status": "ok"}


# ── CLI entrypoint ─────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="vLLM Policy Server")
    parser.add_argument("--model-path", type=str, required=False, help="Path to model checkpoint directory",default="/workspace/rl-max/checkpoints/policy")
    parser.add_argument("--tp", type=int, default=1, help="Tensor parallel size")
    parser.add_argument("--port", type=int, default=8000, help="Server port")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Server host")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    client = VLLMClient(model_path=args.model_path, tensor_parallel_size=args.tp)
    uvicorn.run(app, host=args.host, port=args.port)
