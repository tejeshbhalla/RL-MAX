# rl-max

Minimal GRPO (Group Relative Policy Optimization) training loop for LLMs. Built from scratch with vLLM for fast generation and DeepSpeed ZeRO-2 for memory-efficient training.

## Architecture

```
┌─────────────┐     HTTP      ┌─────────────────┐
│  vLLM Server │◄────────────►│   Orchestrator   │
│  (generation)│   /generate  │   (main loop)    │
└─────────────┘   /reload     │                  │
                               │  ┌────────────┐ │
                               │  │  Trainer    │ │
                               │  │  (DeepSpeed)│ │
                               │  └────────────┘ │
                               │  ┌────────────┐ │
                               │  │  Rewarder   │ │
                               │  └────────────┘ │
                               └─────────────────┘
```

**Per training step:**
1. Orchestrator sends prompts (repeated `group_size` times each) to vLLM
2. vLLM generates completions + old log probs
3. Rewarder scores each completion
4. Trainer computes group-relative advantages, runs K gradient epochs with PPO-style clipped loss
5. Updated weights saved to disk, vLLM hot-reloads them

## Project Structure

```
rl-max/
├── src/
│   ├── vllm_client.py     # vLLM FastAPI server for rollout generation
│   ├── trainer.py          # GRPO trainer with DeepSpeed ZeRO-2
│   ├── orchestrator.py     # Main training loop (generate → reward → train → sync)
│   ├── rewards.py          # Pluggable reward registry + built-in rewards
│   └── data.py             # Dataset loading (WIP)
├── config/
│   ├── ds_config.json      # DeepSpeed config (ZeRO-2, CPU optimizer offload)
│   └── config.yaml         # Training config (WIP)
├── checkpoints/
│   └── policy/             # HuggingFace model checkpoint (shared by vLLM + trainer)
├── run.py                  # Quick API test script
└── requirements.txt
```

## Setup

```bash
# Install dependencies
pip install -r requirements.txt

# Download model (Qwen 2.5 1.5B Instruct)
python3 -c "
from huggingface_hub import snapshot_download
snapshot_download('Qwen/Qwen2.5-1.5B-Instruct', local_dir='checkpoints/policy')
"
```

## Running

### 1. Start vLLM server (Terminal 1)

```bash
# Single GPU
python3 -m src.vllm_client --model-path /workspace/rl-max/checkpoints/policy

# With tensor parallelism
python3 -m src.vllm_client --model-path /workspace/rl-max/checkpoints/policy --tp 2
```

### 2. Run training (Terminal 2)

```bash
# Use a different GPU than vLLM
CUDA_VISIBLE_DEVICES=1 deepspeed --num_gpus 1 -m src.orchestrator
```

If running on a single GPU (vLLM + trainer share it):

```bash
CUDA_VISIBLE_DEVICES=0 deepspeed --num_gpus 1 -m src.orchestrator
```

### 3. Test the API (optional)

```bash
python3 run.py
```

## Key Components

### vLLM Client (`src/vllm_client.py`)

FastAPI server wrapping vLLM for batched generation. Endpoints:
- `POST /generate` — batch generate completions with log probs
- `POST /reload_weights` — hot-reload model weights from disk after training
- `GET /health` — health check

### Trainer (`src/trainer.py`)

GRPO trainer with DeepSpeed ZeRO-2. Key methods:
- `prepare_inputs()` — tokenize prompt+completion, build attention/completion masks
- `compute_token_log_probs()` — forward pass → shifted log-softmax → per-token log probs
- `sequence_log_probs()` — sum token log probs over completion mask
- `compute_advantages()` — group-relative normalization: `(r - mean) / (std + ε)`
- `grpo_loss()` — PPO-style clipped objective at sequence level
- `save_checkpoint()` — save HF-format weights for vLLM reload

### Rewards (`src/rewards.py`)

Pluggable reward system with auto-registration:
- `LengthReward` — rewards longer completions (normalized to [0, 1])
- `ThinkFormatReward` — rewards `<think>...</think>` structure
- Add custom rewards by subclassing `BaseReward` and decorating with `@register_reward`

### Orchestrator (`src/orchestrator.py`)

Main training loop:
1. Repeat each prompt `group_size` times → send to vLLM
2. Score completions with `Rewarder`
3. Run `num_epochs` gradient updates per batch (old log probs frozen, new log probs updated each epoch)
4. Save checkpoint + reload vLLM weights

## Configuration

### DeepSpeed (`config/ds_config.json`)

```json
{
  "bf16": {"enabled": true},
  "zero_optimization": {
      "stage": 2,
      "offload_optimizer": {"device": "cpu", "pin_memory": true}
  },
  "gradient_clipping": 1.0,
  "train_batch_size": 16,
  "train_micro_batch_size_per_gpu": 16
}
```

### Training Hyperparameters

| Parameter | Default | Location |
|-----------|---------|----------|
| Learning rate | 1e-6 | `trainer.py` |
| Clip epsilon | 0.2 | `trainer.py` |
| Group size | 4 | `orchestrator.py` |
| Epochs per step | 4 | `orchestrator.py` |
| Max tokens | 128 | `orchestrator.py` |
| Temperature | 0.7 | `orchestrator.py` |
| Training steps | 10 | `orchestrator.py` |

## GRPO Algorithm Summary

```
For each training step:
    1. Sample prompts P₁, P₂, ..., Pₙ
    2. For each Pᵢ, generate K completions → group of K samples
    3. Score each completion with reward function → rᵢⱼ
    4. Compute group-relative advantage: Aᵢⱼ = (rᵢⱼ - mean(rᵢ)) / (std(rᵢ) + ε)
    5. For each epoch k = 1..K:
        a. Forward pass → new log probs
        b. ratio = exp(new_log_prob - old_log_prob)
        c. loss = -mean(min(ratio × A, clip(ratio, 1±ε) × A))
        d. Backward + optimizer step
    6. Save weights → reload vLLM
```

## Known Limitations

- **Sequence-level loss**: ratios are computed at the sequence level (sum of token log probs), not per-token. Can be unstable for long completions.
- **No KL penalty**: no reference model constraint — policy can drift far from the base model.
- **No micro-batching**: entire batch goes through one forward pass. Large batches or long sequences may OOM.
- **Disk-based weight sync**: checkpoint save + vLLM reload is slow. Production systems use shared memory.
- **Hardcoded prompts**: uses 4 fixed prompts. Needs proper dataset integration.
- **Single-node only**: designed for single-node multi-GPU. No multi-node support.

## Roadmap

- [ ] KL divergence penalty against reference policy (removed as not used now )
- [ ] Token-level clipped loss (more stable)
- [ ] Dataset loader in `data.py`
- [ ] ZeRO-3 checkpoint saving

