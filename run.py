"""
Quick test script for VLLMClient — calls the /generate API endpoint.
Assumes the server is already running (python -m src.vllm_client --model-path ...).
"""

import requests
from transformers import AutoTokenizer

SERVER_URL = "http://localhost:8000"


def main():
    # ── 1. Health check ───────────────────────────────────────────────
    print("Checking server health …")
    resp = requests.get(f"{SERVER_URL}/health")
    resp.raise_for_status()
    print(f"Server status: {resp.json()}\n")

    # ── 2. Build a batch of conversations ─────────────────────────────
    messages = [
        # Conversation 1: simple math
        [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "What is 2 + 2? think step by step."},
        ],
        # Conversation 2: general knowledge
        [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "What is the capital of France? Answer in one sentence."},
        ],
        # Conversation 3: coding question
        [
            {"role": "user", "content": "Write a Python function that returns the factorial of n."},
        ],
    ]

    payload = {
        "messages": messages,
        "max_tokens": 1024,
        "temperature": 0.7,
        "top_p": 1.0,
        "top_k": -1,
    }

    # ── 3. Call /generate ─────────────────────────────────────────────
    print("Calling /generate …")
    resp = requests.post(f"{SERVER_URL}/generate", json=payload)
    resp.raise_for_status()
    data = resp.json()
    
    all_string = data['prompt'][0]+data['completion'][0]
    print(data['old_log_probs'][0])
    tokenizer = AutoTokenizer.from_pretrained('/workspace/rl-max/checkpoints/policy')
    input_ids = tokenizer(all_string,add_special_tokens=False).input_ids
    print(tokenizer.decode(input_ids))

if __name__ == "__main__":
    main()
