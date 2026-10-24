"""GSM8K test subset: 256 questions sampled without replacement (seed 42), zero-shot chat prompts, thinking off."""
import json
import os
import random

from datasets import load_dataset


def load_gsm8k_256(root, tok, n=256, seed=42):
    ds = load_dataset("openai/gsm8k", "main", split="test")
    path = os.path.join(root, "data/gsm8k_256_indices.json")
    if os.path.exists(path):
        idx = json.load(open(path))["indices"]
    else:
        idx = random.Random(seed).sample(range(len(ds)), n)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        json.dump(dict(dataset="openai/gsm8k main test", n_total=len(ds), n=n, seed=seed,
                       method="random.Random(seed).sample(range(n_total), n)", indices=idx), open(path, "w"), indent=1)
    prompts = []
    for i in idx:
        text = tok.apply_chat_template([{"role": "user", "content": ds[i]["question"]}], tokenize=False,
                                       add_generation_prompt=True, enable_thinking=False)
        prompts.append(tok.encode(text, add_special_tokens=False))
    return idx, prompts
