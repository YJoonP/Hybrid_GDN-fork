"""Paired comparison of every condition against a reference condition on the same samples.

usage: compare.py "<glob>" [ref_cond]
Prints acc, delta-acc (wins/losses), exact McNemar p, mean delta gold-answer log-prob with 95% bootstrap CI.
"""
import collections
import glob
import json
import math
import random
import sys


def mcnemar(b, c):
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    p = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2 * p)


def boot_ci(x, it=2000, seed=0):
    rng = random.Random(seed)
    n = len(x)
    ms = sorted(sum(x[rng.randrange(n)] for _ in range(n)) / n for _ in range(it))
    return ms[int(0.025 * it)], ms[int(0.975 * it)]


def main():
    pat = sys.argv[1]
    ref = sys.argv[2] if len(sys.argv) > 2 else "base"
    groups = collections.defaultdict(dict)
    for f in sorted(glob.glob(pat)):
        for l in open(f):
            r = json.loads(l)
            groups[(r["task"], r["ctx"], r["fmt"])].setdefault(r["cond"], {})[r["idx"]] = r
    for key, conds in sorted(groups.items()):
        if ref not in conds:
            continue
        base = conds[ref]
        print(f"\n== {key[0]} ctx={key[1]} fmt={key[2]}  (ref={ref})")
        print(f"{'cond':<46}{'n':>4}{'acc':>7}{'dAcc':>7}{'W/L':>8}{'p':>7}{'dLP':>8}{'95%CI':>17}{'kept':>7}")
        for c, rows in conds.items():
            ids = sorted(set(rows) & set(base))
            if not ids:
                continue
            n = len(ids)
            acc = sum(rows[i]["acc"] for i in ids) / n
            bacc = sum(base[i]["acc"] for i in ids) / n
            w = sum(rows[i]["acc"] > base[i]["acc"] for i in ids)
            lo = sum(rows[i]["acc"] < base[i]["acc"] for i in ids)
            d = [rows[i]["ans_lp"] - base[i]["ans_lp"] for i in ids]
            ci = boot_ci(d) if c != ref else (0, 0)
            kept = sum(rows[i]["kept"] for i in ids) / n
            print(f"{c:<46}{n:>4}{acc:7.3f}{acc - bacc:+7.3f}{f'{w}/{lo}':>8}{mcnemar(w, lo):7.3f}"
                  f"{sum(d) / n:+8.3f}  [{ci[0]:+6.2f},{ci[1]:+6.2f}]{kept:7.3f}")


if __name__ == "__main__":
    main()
