"""Integration check of conf-K method B (confk-head-B) in the CUDA-graph engine.

1. Head reimplementation: dflash_cb.rollback's head (load_conf_head + _ConfHead.confidence) on the package's sample
   rounds (sample/hid, sample/embed_rows) must reproduce sample/expected_conf.npz (linear and mlp).
2. Engine inputs: GSM8K test questions 1 and 4 (0-based) run through GraphEngine (batch 1, confkb, mlp head). For
   every decode round whose verify-block tokens (prev) equal the package's, compare our draft hidden h and head
   confidence c with the package's (h from their engine, c from expected_conf).
Gate: 1 exact (< 1e-5); round 1 of both questions starts from the same anchor; on every compared round the draft
hidden matches (min cosine > 0.999), c matches where the verify tokens agree (< 0.02), and every differing draft token
sits where their draft's top-1 / top-2 margin is < 0.2 (a bf16 near-tie). The draft input is the anchor plus mask
tokens, so h does not depend on draft tokens; rounds are compared while the committed history agrees. Writes results_graph_confkb_check.json.
"""
import json
import os
import sys

import numpy as np
import torch
from datasets import load_dataset
from dflash.model import DFlashDraftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from dflash_cb.graph_engine import GraphEngine
from dflash_cb.rollback import _ConfHead, load_conf_head

ROOT = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.join(ROOT, "ext/confk_head_B/confk-head-B")
QUESTIONS = {"gsm8k_test-0001": 1, "gsm8k_test-0004": 4}


class _Stub(_ConfHead):
    """_ConfHead.confidence with a small embedding table (the sample's rows only)."""
    def __init__(self, head, E):
        self.head = head
        self.eng = type("E", (), {"t": type("T", (), {"get_input_embeddings": lambda s: type("W", (), {"weight": E})()})()})()


def check_head():
    exp = np.load(os.path.join(PKG, "sample/expected_conf.npz"))
    er = np.load(os.path.join(PKG, "sample/embed_rows.npz"))
    E, remap = torch.from_numpy(er["E"]), {int(t): i for i, t in enumerate(er["tokens"])}
    out = {}
    for kind in ("linear", "mlp"):
        stub = _Stub(load_conf_head(os.path.join(PKG, f"heads/head_{kind}.pt"), "cpu"), E)
        for q in QUESTIONS:
            z = np.load(os.path.join(PKG, f"sample/hid/{q}.npz"))
            prev = torch.tensor(np.vectorize(remap.get)(z["prev"]))
            c = stub.confidence(torch.from_numpy(z["hid"]), prev).numpy()
            out[f"{kind}/{q}"] = float(np.abs(c - exp[f"{kind}/{q}"]).max())
    return out


def check_engine():
    dev = torch.device("cuda:0")
    tok = AutoTokenizer.from_pretrained(os.path.join(ROOT, "models/Qwen3.5-4B"))
    target = AutoModelForCausalLM.from_pretrained(os.path.join(ROOT, "models/Qwen3.5-4B"), attn_implementation="sdpa",
                                                  dtype=torch.bfloat16).to(dev).eval()
    draft = DFlashDraftModel.from_pretrained(os.path.join(ROOT, "models/Qwen3.5-4B-DFlash"), attn_implementation="sdpa",
                                             dtype=torch.bfloat16).to(dev).eval()
    ds = load_dataset("openai/gsm8k", "main", split="test")
    exp = np.load(os.path.join(PKG, "sample/expected_conf.npz"))
    rounds = [json.loads(l) for l in open(os.path.join(PKG, "sample/rounds_sample.jsonl"))]
    head = os.path.join(PKG, "heads/head_mlp.pt")
    prompts = {q: tok.encode(tok.apply_chat_template([{"role": "user", "content": ds[i]["question"]}], tokenize=False,
                                                     add_generation_prompt=True, enable_thinking=False),
                             add_special_tokens=False) for q, i in QUESTIONS.items()}
    eng = GraphEngine(target, draft, "dflash", 1, max(len(p) for p in prompts.values()), 128, 16, "confkb",
                      dict(select="topk", k=1, thr=0.5, head=head))
    eng.capture()
    res = {}
    for q, ids in prompts.items():
        z = np.load(os.path.join(PKG, f"sample/hid/{q}.npz"))
        ref_acc = [r["n_acc"] for r in rounds if r["prompt"] == q]
        eng.insert(q, torch.tensor(ids))
        per = []
        for r in range(len(z["prev"])):
            _, prod, fin, _ = eng.step()
            h, prev, c = (x[0].float().cpu().numpy() for x in eng.rb.last)
            ref = next(x for x in rounds if x["prompt"] == q and x["round"] == r + 1)
            same = prev == z["prev"][r]
            row = dict(round=r + 1, anchor_identical=bool(same[0]), prev_identical=bool(same.all()),
                       n_acc=prod[0] - 1, ref_n_acc=ref_acc[r])
            if same[0]:             # same context and anchor -> same draft input; h does not depend on draft tokens
                hr = z["hid"][r].astype(np.float32)
                cos = (h * hr).sum(-1) / (np.linalg.norm(h, axis=-1) * np.linalg.norm(hr, axis=-1))
                # draft token d_j (j = 1..14) is prev[j]; their draft margin at that position is p1 - p2 of draft j
                mism = [dict(pos=j, p1=ref["p1"][j - 1], margin=ref["p1"][j - 1] - ref["p2"][j - 1])
                        for j in range(1, 15) if not same[j]]
                row.update(h_max_abs=float(np.abs(h - hr).max()), h_min_cos=float(cos.min()),
                           c_max_abs_same_prev=float(np.abs(c - exp[f"mlp/{q}"][r])[same].max()),
                           token_mismatches=mism)
            per.append(row)
            if not same[0] or row["n_acc"] != row["ref_n_acc"] or fin:
                break
        eng.reset()
        res[q] = dict(prompt_len=len(ids), rounds=per)
    return res


def main():
    head = check_head()
    print("[head] max |c - expected|:", head, flush=True)
    eng = check_engine()
    for q, r in eng.items():
        n_same = sum(x["anchor_identical"] for x in r["rounds"])
        print(f"[engine] {q}: prompt {r['prompt_len']} tok, rounds compared {n_same}/{len(r['rounds'])}")
        for x in r["rounds"]:
            print("   ", x)
    rows = [x for r in eng.values() for x in r["rounds"] if x["anchor_identical"]]
    ok = (all(v < 1e-5 for v in head.values()) and all(r["rounds"][0]["anchor_identical"] for r in eng.values())
          and all(x["h_min_cos"] > 0.999 and x["c_max_abs_same_prev"] < 0.02 for x in rows)
          and all(m["margin"] < 0.2 for x in rows for m in x["token_mismatches"]))
    json.dump(dict(head_reimpl=head, engine=eng, passed=ok), open(os.path.join(ROOT, "results_graph_confkb_check.json"),
                                                                    "w"), indent=1)
    print("[check] passed" if ok else "[check] FAILED", flush=True)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
