import requests
from src.trainer import GrpoTrainer
from src.rewards import Rewarder

VLLM_URL = "http://localhost:8000"
MODEL_PATH = "/workspace/rl-max/checkpoints/policy"
DS_CONFIG = "/workspace/rl-max/config/ds_config.json"

# ── Hardcoded prompts for MVP ────────────────────────────────────────
PROMPTS = [
    "What is 2 + 2? Think step by step.",
    "What is the capital of France? Think step by step.",
    "Write a Python function that reverses a string. Think step by step.",
    "Explain gravity in one sentence. Think step by step.",
]


class Orchestrator:
    def __init__(
        self,
        group_size: int = 4,
        max_tokens: int = 256,
        num_steps: int = 10,
        num_epochs: int = 4,
    ):
        self.group_size = group_size
        self.max_tokens = max_tokens
        self.num_steps = num_steps
        self.num_epochs = num_epochs
        self.trainer = GrpoTrainer(MODEL_PATH, DS_CONFIG, group_size=group_size)
        self.rewarder = Rewarder()

    def generate_rollouts(self, messages_batch: list[list[dict[str, str]]]) -> dict:
        payload = {
            "messages": messages_batch,
            "max_tokens": self.max_tokens,
            "temperature": 0.7,
            "top_p": 1.0,
            "top_k": -1,
        }
        resp = requests.post(f"{VLLM_URL}/generate", json=payload)
        resp.raise_for_status()
        return resp.json()

    def reload_vllm_weights(self):
        resp = requests.post(f"{VLLM_URL}/reload_weights")
        resp.raise_for_status()

    def run(self):
        for step in range(self.num_steps):
            print(f"\n{'='*60}")
            print(f"Step {step + 1}/{self.num_steps}")
            print(f"{'='*60}")

            # 1. Build messages: repeat each prompt group_size times
            messages_batch = []
            for prompt_text in PROMPTS:
                msg = [{"role": "user", "content": prompt_text}]
                for _ in range(self.group_size):
                    messages_batch.append(msg)

            # 2. Generate rollouts from vLLM (old policy)
            rollout = self.generate_rollouts(messages_batch)
            prompts = rollout["prompt"]
            completions = rollout["completion"]
            old_log_probs = rollout["old_log_probs"]

            # 3. Score with rewards
            metadata = [{} for _ in prompts]
            rewards = self.rewarder.score_batch(prompts, completions, metadata)

            # 4. Prepare batch once (shared across epochs)
            batch = self.trainer.prepare_inputs(prompts, completions)
            input_ids = batch["input_ids"]
            attention_mask = batch["attention_mask"]
            completion_mask = batch["completion_mask"]

            import torch
            old_seq_lp = torch.tensor(
                [sum(lp) for lp in old_log_probs],
                dtype=torch.float, device=input_ids.device,
            )
            rewards_t = torch.tensor(rewards, dtype=torch.float, device=input_ids.device)
            advantages = self.trainer.compute_advantages(rewards_t)

            # 5. Multiple epochs on same batch
            for epoch in range(self.num_epochs):
                new_token_lp = self.trainer.compute_token_log_probs(input_ids, attention_mask)
                new_seq_lp = self.trainer.sequence_log_probs(new_token_lp, completion_mask)

                loss = self.trainer.grpo_loss(new_seq_lp, old_seq_lp, advantages)
                self.trainer.model_engine.backward(loss)
                self.trainer.model_engine.step()

                print(f"  epoch {epoch+1}/{self.num_epochs} | loss: {loss.detach().item():.4f}")

            # 6. Log step summary
            print(f"  reward_mean: {rewards_t.mean().item():.4f}")
            print(f"  reward_std:  {rewards_t.std(unbiased=False).item():.4f}")
            print(f"  adv_mean:    {advantages.mean().item():.4f}")
            print(f"  adv_std:     {advantages.std(unbiased=False).item():.4f}")
            print(f"  sample: {completions[0][:120]}...")

            # 7. Sync weights to vLLM
            # TODO: save checkpoint + reload
            self.trainer.save_checkpoint()
            self.reload_vllm_weights()

        print("\nTraining complete!")


if __name__ == "__main__":
    orch = Orchestrator(group_size=4, max_tokens=128, num_steps=10, num_epochs=4)
    orch.run()