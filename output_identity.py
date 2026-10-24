"""Output identity across rollback strategies, runs and engines (eager vs CUDA graph).

For each batch size: how many of the 256 questions produce the same tokens in every DFlash run of an engine (all
strategies, all runs), and how eager and graph outputs compare (DFlash vs DFlash, AR vs AR, DFlash vs AR).
Writes results_compare_graph/output_identity.{json,md}.
"""
import glob
import json
import os

ROOT = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(ROOT, "results_compare_graph")


def load(path):
    return {r["qid"]: tuple(r["output"]) for r in map(json.loads, open(path))}


def runs(engine, kind, bs):
    pat = os.path.join(ROOT, "results_graph_*" if engine == "graph" else "results_*", "raw", f"{kind}_bs{bs}_run*.jsonl")
    out = {}
    for p in sorted(glob.glob(pat)):
        d = p.split(os.sep)[-3]
        if p.endswith("_steps.jsonl"):
            continue
        if engine == "eager" and (d.startswith("results_graph_") or d.startswith("results_compare")):
            continue
        out[f"{d}/{os.path.basename(p)[:-6]}"] = load(p)
    return out


def agree(group):
    """Questions on which every run in the group gives the same output."""
    qids = set.intersection(*(set(r) for r in group.values()))
    return sum(len({r[q] for r in group.values()}) == 1 for q in qids), len(qids)


def cross(a, b):
    """Questions where run a and run b agree (first run of each group)."""
    ra, rb = next(iter(a.values())), next(iter(b.values()))
    qids = set(ra) & set(rb)
    return sum(ra[q] == rb[q] for q in qids), len(qids)


def main():
    res, md = {}, ["| batch | eager DFlash 실행 수 | eager 전부 동일 | graph DFlash 실행 수 | graph 전부 동일 | "
                   "DFlash eager=graph | AR eager=graph | graph DFlash=graph AR |",
                   "|---|---|---|---|---|---|---|---|"]
    for bs in (1, 8, 64, 128):
        e, g = runs("eager", "dflash", bs), runs("graph", "dflash", bs)
        ea, ga = runs("eager", "ar", bs), runs("graph", "ar", bs)
        row = dict(eager_runs=len(e), graph_runs=len(g))
        if e:
            row["eager_all_same"] = agree(e)
        if g:
            row["graph_all_same"] = agree(g)
            row["graph_run_dirs"] = sorted({k.split("/")[0] for k in g})
        if e and g:
            row["dflash_eager_vs_graph"] = cross(e, g)
            row["dflash_eager_vs_graph_union"] = agree({**e, **g})
        if ea and ga:
            row["ar_eager_vs_graph"] = cross(ea, ga)
        if g and ga:
            row["graph_dflash_vs_ar"] = cross(g, ga)
        res[bs] = row
        f = lambda k: f"{row[k][0]}/{row[k][1]}" if k in row else "–"
        md.append(f"| {bs} | {len(e)} | {f('eager_all_same')} | {len(g)} | {f('graph_all_same')} | "
                  f"{f('dflash_eager_vs_graph')} | {f('ar_eager_vs_graph')} | {f('graph_dflash_vs_ar')} |")
    os.makedirs(OUT, exist_ok=True)
    json.dump(res, open(os.path.join(OUT, "output_identity.json"), "w"), indent=1)
    open(os.path.join(OUT, "output_identity.md"), "w").write("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
