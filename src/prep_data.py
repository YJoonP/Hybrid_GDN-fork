"""Download HotpotQA (distractor, validation) and build:
  data/hotpot_val.jsonl   : bridge/comparison questions with non yes/no answers
  data/para_pool.json     : deduplicated Wikipedia paragraphs (haystack / extra distractors)
"""
import json
import os
import sys

from datasets import load_dataset
from transformers import AutoTokenizer

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
tok = AutoTokenizer.from_pretrained(os.path.join(ROOT, "models/Qwen3.5-4B"))
ds = load_dataset("hotpotqa/hotpot_qa", "distractor", split="validation")

pool, seen = [], set()
with open(os.path.join(ROOT, "data/hotpot_val.jsonl"), "w") as f:
    for ex in ds:
        for title, sents in zip(ex["context"]["title"], ex["context"]["sentences"]):
            if title in seen:
                continue
            seen.add(title)
            text = f"Title: {title}\n" + "".join(sents)
            pool.append(dict(title=title, text=text))
        if ex["answer"].lower() in ("yes", "no"):
            continue
        f.write(json.dumps(dict(id=ex["id"], question=ex["question"], answer=ex["answer"], type=ex["type"],
                                level=ex["level"], supporting_facts=ex["supporting_facts"],
                                context=ex["context"])) + "\n")

ntok = tok([p["text"] for p in pool]).input_ids
for p, n in zip(pool, ntok):
    p["ntok"] = len(n)
pool = [p for p in pool if 30 <= p["ntok"] <= 400]
with open(os.path.join(ROOT, "data/para_pool.json"), "w") as f:
    json.dump(pool, f)
print("pool", len(pool), "avg tok", sum(p["ntok"] for p in pool) / len(pool), file=sys.stderr)
