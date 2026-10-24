"""Cross-method comparison of every results dir -> results_compare/.

    python analyze_all.py

Cells ran on two physical RTX 3090s, so speedups use the AR cell measured on the same GPU
(results_fullstate, plus results_ar_gpu1 when a second GPU was used; mapped by the GPU recorded in each AR cell). Writes per-dir summary.csv / plots / report_data.md and, in
results_compare/: summary_all.csv, table_all.md, table_speedup.md, table_gpu0.md, gpu_calibration.md,
rollback_stats.md, best_confk.json, plots/*.png.
"""
import argparse
import json
import os
import statistics as st

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from dflash_cb import report as R  # noqa: E402

ROOT = os.path.dirname(os.path.abspath(__file__))
_ap = argparse.ArgumentParser()
_ap.add_argument("--engine", default="eager", choices=["eager", "graph"])
ENGINE = _ap.parse_known_args()[0].engine
PREFIX = "results_graph_" if ENGINE == "graph" else "results_"
OUT = os.path.join(ROOT, "results_compare_graph" if ENGINE == "graph" else "results_compare")
def _ar_dirs():
    """{gpu index: results dir holding AR cells measured on that GPU} from the cells' own records."""
    out = {}
    for d in ((PREFIX + "fullstate",) if ENGINE == "graph" else ("results_fullstate", "results_ar_gpu1")):
        p = os.path.join(ROOT, d, "cells", "ar_bs1.json")
        if os.path.exists(p) and not json.load(open(p)).get("reused_from"):
            out.setdefault(R.gpu_of(json.load(open(p))), os.path.join(ROOT, d))
    return out


AR = _ar_dirs()
CONFK_V = ("top1", "top2", "top4", "thr0.1", "thr0.3", "thr0.5", "thr0.7", "thr0.9")
if ENGINE == "graph":       # one GPU, one directory per strategy
    DIRS = {"fullstate": PREFIX + "fullstate", "modeal_gpu0": PREFIX + "modeal", "specla_gpu0": PREFIX + "specla",
            **{f"confk_{v}": f"{PREFIX}confk_{v}" for v in CONFK_V},
            **{f"confkb_{v}": f"{PREFIX}confkb_{v}" for v in CONFK_V},       # conf-K method B (confidence head)
            "oracle": PREFIX + "oracle"}                                       # true AL from a recorded run
else:      # key -> results dir name (key also selects label/colour)
    DIRS = {"fullstate": "results_fullstate", "modeal": "results_modeal", "specla": "results_specla",
            "modeal_gpu0": "results_modeal_gpu0", "specla_gpu0": "results_specla_gpu0",
            **{f"confk_{v}": f"results_confk_{v}" for v in CONFK_V}}
R.LABELS.update({"modeal_gpu0": R.LABELS["modeal"], "specla_gpu0": R.LABELS["specla"]})
R.COLORS.update({"modeal_gpu0": R.COLORS["modeal"], "specla_gpu0": R.COLORS["specla"]})


def rows_for(key):
    d = os.path.join(ROOT, DIRS[key])
    if not os.path.isdir(os.path.join(d, "cells")):
        return None
    rows, _ = R.cell_rows(d, key.replace("_gpu0", ""))
    rows = [dict(r, key=key) for r in rows if r["key"] != "ar"]
    return R.add_relative(rows, d, AR)


def ar_rows(gpu):
    d = AR.get(gpu)
    if not d or not os.path.isdir(os.path.join(d, "cells")):
        return None
    rows, _ = R.cell_rows(d, "fullstate")
    return R.add_relative([dict(r, key="ar") for r in rows if r["key"] == "ar"], d, AR)


def stats_rows(key):
    """confk: hit / from-S0 / replay statistics averaged over runs; modeal: chosen m."""
    out = []
    d = os.path.join(ROOT, DIRS[key], "cells")
    for bs in R.BATCHES:
        p = os.path.join(d, f"dflash_bs{bs}.json")
        runs = json.load(open(p)).get("runs", []) if os.path.exists(p) else []
        if runs and "prediction" in runs[0]:
            # graph runs recorded before the in-place reset_stats fix have an empty run 2 (counters lost their
            # captured address); the statistics are deterministic, so the runs with counts are used
            pr = [r["prediction"] for r in runs if r["prediction"]["row_steps"] > 0]
            out.append((bs, dict(slots=pr[0]["slots_per_seq"], hit=st.mean(x["hit_rate"] for x in pr),
                                 s0=st.mean(x["from_s0_rate"] for x in pr),
                                 replay=st.mean(x["replay_tokens_per_row_step"] for x in pr))))
        elif runs and "m_counts" in runs[0]:
            mc = runs[0]["m_counts"]
            out.append((bs, dict(m=max(mc, key=lambda m: mc[m]), m_counts=mc)))
    return out


def speedup_table(series):
    bss = R.BATCHES
    lines = ["| 방식 | GPU | " + " | ".join(f"batch {b}" for b in bss) + " |", "|---|---|" + "---|" * len(bss)]
    for k, rows in series.items():
        by = {r["batch"]: r for r in rows}
        cells = []
        for b in bss:
            r = by.get(b, {})
            if r.get("speedup") is not None:
                cells.append(f"{r['speedup']:.2f}× ({r['throughput']:.0f})")
            else:
                cells.append("OOM" if str(r.get("status", "")).startswith("oom") else "–")
        gpu = next((r.get("gpu") for r in rows if r.get("gpu")), "–")
        lines.append(f"| {R.lab(k)} | {gpu} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def best_confk(confk):
    common = [b for b in R.BATCHES if all(any(r["batch"] == b and r.get("speedup") for r in v) for v in confk.values())]
    if not common:
        return None, common, {}
    score = {k: st.mean(next(r["speedup"] for r in v if r["batch"] == b) for b in common) for k, v in confk.items()}
    return max(score, key=score.get), common, score


def confk_panels(confk, refs, path, confkb=None):
    """Dot plot per batch size: each conf-K variant's speedup (mean ± std), other methods as labelled lines. With
    confkb (method B, same variants), its dots sit to the right of method A's."""
    confkb = confkb or {}
    bss = [b for b in R.BATCHES if any(r.get("speedup") for v in confk.values() for r in v if r["batch"] == b)]
    if not bss:
        return
    keys = list(confk)
    dx = 0.17 if confkb else 0.0
    fig, axes = plt.subplots(1, len(bss), figsize=((6.4 if confkb else 4.8) * len(bss), 3.9), dpi=150, squeeze=False)
    for ax, b in zip(axes[0], bss):
        vals = []
        for x, k in enumerate(keys):
            r = next((r for r in confk[k] if r["batch"] == b), {})
            if r.get("speedup") is None:
                vals.append(None)
                continue
            vals.append(r["speedup"])
            ax.errorbar(x - dx, r["speedup"], yerr=r["speedup_std"], fmt="o", ms=6, color=R.col("confk"),
                        ecolor=R.MUTED, elinewidth=0.8, capsize=2,
                        label="method A (softmax p1)" if (confkb and x == 0 and b == bss[0]) else None)
            if not confkb:          # with A and B side by side the values are in the table instead
                ax.annotate(f"{r['speedup']:.2f}", (x - dx, r["speedup"]), textcoords="offset points",
                            xytext=(0, 7 if x % 2 == 0 else -12), ha="center", fontsize=7, color=R.INK)
        for x, k in enumerate(keys):
            kb = k.replace("confk_", "confkb_")
            r = next((r for r in confkb.get(kb, []) if r["batch"] == b), {})
            if r.get("speedup") is None:
                continue
            vals.append(r["speedup"])
            ax.errorbar(x + dx, r["speedup"], yerr=r["speedup_std"], fmt="s", ms=5, color=R.col("confkb"),
                        ecolor=R.MUTED, elinewidth=0.8, capsize=2,
                        label="method B (confidence head)" if (x == 0 and b == bss[0]) else None)
        lines = sorted(((r["speedup"], k) for k, rows in refs.items()
                        for r in rows if r["batch"] == b and r.get("speedup") is not None), reverse=True)
        allv = [v for v in vals if v is not None] + [v for v, _ in lines]
        span = (max(allv) - min(allv)) or 0.1
        ax.set_ylim(min(allv) - 0.25 * span, max(allv) + 0.35 * span)
        for x, v in enumerate(vals[:len(keys)]):
            if v is None:
                ax.text(x, ax.get_ylim()[0], " OOM", ha="center", va="bottom", fontsize=7, color=R.MUTED, rotation=90)
        last = None
        for v, k in lines:                        # labels right of the axes, nudged apart when lines are close
            ax.axhline(v, color=R.col(k), lw=1.6)
            yl = v if last is None else min(v, last - 0.08 * span)
            ax.annotate(R.lab(k).replace("DFlash-b16 ", "") + f" {v:.2f}", (1.0, yl), xycoords=("axes fraction", "data"),
                        xytext=(4, 0), textcoords="offset points", va="center", fontsize=7, color=R.INK)
            last = yl
        gpu = {k: next((r.get("gpu") for r in confk[k] if r.get("gpu")), "?") for k in keys}
        one_gpu = len(set(gpu.values())) == 1
        ax.set_xticks(range(len(keys)), [k.replace('confk_', '') + ("" if one_gpu else f" (G{gpu[k]})") for k in keys],
                      rotation=45, fontsize=7)
        ax.set_xlim(-0.5, len(keys) - 0.5)
        R._style(ax, f"batch {b}", "speedup over same-GPU AR (×)")
        ax.set_xlabel("conf-K variant", color=R.MUTED, fontsize=9)
    fig.suptitle("conf-K variants (dots, mean ± std of 3 runs" + ("" if ENGINE == "graph" else "; G0/G1 = GPU measured on")
                 + f") vs GPU-0 methods (lines), {ENGINE} engine", color=R.INK, fontsize=11, x=0.01, ha="left")
    if confkb:
        fig.legend(loc="lower center", ncol=2, frameon=False, fontsize=8, bbox_to_anchor=(0.5, -0.06))
    fig.tight_layout(w_pad=9)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def step_breakdown(series, path, batches=(1, 8, 64)):
    """Stacked bars: wall ms per decode step by section (mean of 3 runs), one panel per batch size."""
    secs = [("draft", "draft"), ("verify", "verify"), ("save", "state save"), ("restore", "state restore"),
            ("other", "accept / KV truncate / slot moves")]
    colors = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]        # validated categorical slots 1-5
    bss = [b for b in batches if any(r["batch"] == b and r.get("steps") for v in series.values() for r in v)]
    fig, axes = plt.subplots(1, len(bss), figsize=(4.6 * len(bss), 3.6), dpi=150, squeeze=False)
    for ax, b in zip(axes[0], bss):
        keys = [k for k, v in series.items() if any(r["batch"] == b and r.get("steps") for r in v)]
        for y, k in enumerate(keys):
            r = next(r for r in series[k] if r["batch"] == b)
            per = {s: 1e3 * (r.get("t_" + s) or 0) / r["steps"] for s, _ in secs[:4]}
            per["other"] = 1e3 * sum(r.get("t_" + s) or 0 for s in ("accept", "kv_truncate", "remove", "host_gap")) \
                / r["steps"]
            left = 0.0
            for (s, name), c in zip(secs, colors):
                ax.barh(y, per[s], left=left, color=c, edgecolor="white", linewidth=1, height=0.6,
                        label=name if (y == 0 and b == bss[0]) else None)
                left += per[s]
            ax.text(left, y, f" {left:.1f}", va="center", fontsize=7, color=R.INK)
        ax.set_yticks(range(len(keys)), [R.lab(k).replace("DFlash-b16 ", "").strip("()") for k in keys], fontsize=7)
        ax.invert_yaxis()
        R._style(ax, f"batch {b}", "")
        ax.set_xlabel("wall ms per decode step (mean of 3 runs)", color=R.MUTED, fontsize=8)
        ax.grid(axis="x", color=R.GRID, linewidth=0.8)
        ax.grid(axis="y", visible=False)
    fig.legend(loc="lower center", ncol=5, frameon=False, fontsize=8, bbox_to_anchor=(0.5, -0.04))
    fig.suptitle(f"Decode step time by section (GPU 0, {ENGINE} engine)", color=R.INK, fontsize=11, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def step_sections(series, path, batches=(1, 8, 64)):
    """Grid of vertical bar charts: rows = batch size, columns = step section (ms per decode step, mean of 3 runs),
    one bar per strategy in a fixed colour; each panel has its own y-scale; OOM cells are named."""
    secs = [("draft", ("draft",)), ("verify", ("verify",)), ("state save", ("save",)), ("restore", ("restore",)),
            ("accept / KV truncate / slot moves", ("accept", "kv_truncate", "remove", "host_gap"))]
    keys = list(series)
    bss = [b for b in batches if any(r["batch"] == b and r.get("steps") for v in series.values() for r in v)]
    fig, axes = plt.subplots(len(bss), len(secs), figsize=(3.4 * len(secs), 2.7 * len(bss)), dpi=150, squeeze=False)
    for i, b in enumerate(bss):
        for j, (name, fields) in enumerate(secs):
            ax = axes[i][j]
            top = 0.0
            for x, k in enumerate(keys):
                r = next((r for r in series[k] if r["batch"] == b), {})
                if not r.get("steps"):
                    ax.text(x, 0, "OOM", rotation=90, ha="center", va="bottom", fontsize=6, color=R.MUTED)
                    continue
                v = 1e3 * sum(r.get("t_" + f) or 0 for f in fields) / r["steps"]
                top = max(top, v)
                ax.bar(x, v, width=0.72, color=R.col(k), label=R.lab(k) if (i == 0 and j == 0) else None)
                ax.annotate(f"{v:.2f}" if v < 10 else f"{v:.1f}", (x, v), xytext=(0, 2), textcoords="offset points",
                            ha="center", va="bottom", fontsize=5.5, color=R.INK)
            ax.set_ylim(0, (top or 1) * 1.18)
            ax.set_xlim(-0.6, len(keys) - 0.4)
            ax.set_xticks([])
            for sp in ("top", "right"):
                ax.spines[sp].set_visible(False)
            for sp in ("left", "bottom"):
                ax.spines[sp].set_color(R.GRID)
            ax.tick_params(colors=R.MUTED, labelsize=7)
            ax.grid(axis="y", color=R.GRID, linewidth=0.8)
            ax.set_axisbelow(True)
            if i == 0:
                ax.set_title(name, color=R.INK, fontsize=9, loc="left")
            if j == 0:
                ax.set_ylabel(f"batch {b}\nms per step", color=R.MUTED, fontsize=8)
    fig.subplots_adjust(left=0.06, right=0.99, top=0.9, bottom=0.13, wspace=0.28, hspace=0.22)
    fig.legend(loc="lower center", ncol=3, frameon=False, fontsize=8, bbox_to_anchor=(0.5, 0.0))
    fig.suptitle(f"Decode step time by section, per strategy (GPU 0, {ENGINE} engine, mean of 3 runs; "
                 "each panel has its own y-scale)", color=R.INK, fontsize=10, x=0.01, ha="left")
    fig.text(0.01, 0.935, "Snapshot writes happen inside the verify kernel and are counted in verify; "
             "'state save' is only the separate copies (e.g. conv windows).", color=R.MUTED, fontsize=8)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def delta_vs_specla(series, base, path, batches=(1, 8, 64)):
    """Where each strategy gains or loses against SpecLA: per section, (strategy - SpecLA) in ms per decode step
    (mean of 3 runs; + = slower than SpecLA), with the whole-step difference last. One panel per batch size."""
    secs = [("draft", ("draft",)), ("verify\n(incl. snapshot writes)", ("verify",)), ("state save", ("save",)),
            ("restore", ("restore",)), ("accept / truncate /\nslot moves / host", ("accept", "kv_truncate", "remove",
                                                                                  "host_gap"))]
    keys = list(series)
    bss = [b for b in batches if any(r["batch"] == b and r.get("steps") for r in base)]
    fig, axes = plt.subplots(len(bss), 1, figsize=(10, 2.9 * len(bss)), dpi=150, squeeze=False)
    w = 0.8 / len(keys)
    for ax, b in zip(axes[:, 0], bss):
        rb = next(r for r in base if r["batch"] == b)
        per = lambda r, f: 1e3 * sum(r.get("t_" + x) or 0 for x in f) / r["steps"]  # noqa: E731
        for j, k in enumerate(keys):
            r = next((r for r in series[k] if r["batch"] == b), {})
            xs = [i + (j - (len(keys) - 1) / 2) * w for i in range(len(secs) + 1)]
            if not r.get("steps"):
                ax.text(xs[-1], 0, "OOM", rotation=90, ha="center", va="bottom", fontsize=6, color=R.MUTED)
                continue
            d = [per(r, f) - per(rb, f) for _, f in secs]
            d.append(1e3 * (r["decode_s"] / r["steps"] - rb["decode_s"] / rb["steps"]))
            ax.bar(xs, d, w * 0.9, color=R.col(k), label=R.lab(k) if b == bss[0] else None)
            for i in (1, len(secs)):                             # label verify and the whole step only
                ax.annotate(f"{d[i]:+.2f}", (xs[i], d[i]), xytext=(0, 2 if d[i] >= 0 else -2),
                            textcoords="offset points", ha="center", va="bottom" if d[i] >= 0 else "top",
                            fontsize=5.5, color=R.INK, rotation=90)
        ax.axhline(0, color=R.INK, linewidth=0.8)
        ax.axvspan(len(secs) - 0.5, len(secs) + 0.5, color=R.GRID, alpha=0.35, zorder=0)
        ax.set_xticks(range(len(secs) + 1), [n for n, _ in secs] + ["whole step"], fontsize=7)
        R._style(ax, f"batch {b}", "ms per step vs SpecLA\n(+ = slower)")
        ax.set_xlabel("")
        lo, hi = ax.get_ylim()
        ax.set_ylim(min(lo, -0.05 * hi) * 1.3, hi * 1.3)
    fig.legend(loc="lower center", ncol=3, frameon=False, fontsize=8, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle(f"Where each strategy loses to SpecLA, per step section (GPU 0, {ENGINE} engine, mean of 3 runs)",
                 color=R.INK, fontsize=10, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


PPT_NAMES = {"oracle": "oracle (true AL, 1 snapshot)", "modeal_gpu0": "modeal (mode AL, 1 snapshot)",
             "confk_thr0.3": "conf-K A (softmax, θ=0.3)", "confkb_top1": "conf-K B (head, top-1)",
             "fullstate": "fullstate (16 snapshots)"}


SLOTMAP_MS = 0.16      # modeal builds its slot map inside verify for each of the 24 GDN layers; oracle once in draft


def oracle_estimate(oracle, specla, modeal):
    """Estimated oracle rows (the oracle was measured ~6 h after SpecLA / modeal; identical draft work ran 1.1-2.0%
    slower then). draft = SpecLA's (same kernels); verify = modeal's minus the slot-map build that modeal does inside
    verify (SLOTMAP_MS per step, from batch 1); other sections = measured / session slowdown, where the slowdown is the
    oracle's draft time over the mean draft time of SpecLA and modeal; the step total = SpecLA's step + the section
    differences."""
    out = []
    for r in oracle:
        sp = next((x for x in specla if x["batch"] == r["batch"]), {})
        md = next((x for x in modeal if x["batch"] == r["batch"]), {})
        if not (r.get("steps") and sp.get("steps") and md.get("steps")):
            out.append(r)
            continue
        f = r["t_draft"] / ((sp["t_draft"] + md["t_draft"]) / 2)
        e = dict(r, estimated=True, session_slowdown=f)
        e["t_draft"] = sp["t_draft"]
        e["t_verify"] = md["t_verify"] - SLOTMAP_MS / 1e3 * r["steps"]
        for k in ("save", "restore", "accept", "kv_truncate", "remove", "host_gap"):
            e["t_" + k] = (r.get("t_" + k) or 0) / f
        secs = ("draft", "verify", "save", "restore", "accept", "kv_truncate", "remove", "host_gap")
        e["decode_s"] = sp["decode_s"] + sum((e["t_" + k] or 0) - (sp.get("t_" + k) or 0) for k in secs)
        e["throughput"] = r["throughput"] * r["decode_s"] / e["decode_s"]
        out.append(e)
    return out


def delta_vs_specla_ppt(series, base, outdir, batches=(1, 8, 64), suffix="", note=""):
    """delta_vs_specla, one slide-sized figure per batch size (16:9, large fonts), saved as 300-dpi PNG and SVG."""
    secs = [("draft", ("draft",)), ("verify\n(incl. snapshot writes)", ("verify",)), ("state save", ("save",)),
            ("restore", ("restore",)), ("accept / truncate /\nslot moves / host", ("accept", "kv_truncate", "remove",
                                                                                  "host_gap"))]
    keys = list(series)
    w = 0.8 / len(keys)
    per = lambda r, f: 1e3 * sum(r.get("t_" + x) or 0 for x in f) / r["steps"]  # noqa: E731
    os.makedirs(outdir, exist_ok=True)
    for b in batches:
        rb = next((r for r in base if r["batch"] == b and r.get("steps")), None)
        if rb is None:
            continue
        fig, ax = plt.subplots(figsize=(13.33, 7.5), dpi=300)
        for j, k in enumerate(keys):
            r = next((r for r in series[k] if r["batch"] == b), {})
            xs = [i + (j - (len(keys) - 1) / 2) * w for i in range(len(secs) + 1)]
            name = PPT_NAMES.get(k, R.lab(k))
            if not r.get("steps"):
                ax.bar([0], [0], color=R.col(k), label=name + "  — OOM")
                ax.text(xs[-1], 0, " OOM", rotation=90, ha="center", va="bottom", fontsize=10, color=R.MUTED)
                continue
            d = [per(r, f) - per(rb, f) for _, f in secs]
            d.append(1e3 * (r["decode_s"] / r["steps"] - rb["decode_s"] / rb["steps"]))
            est = any(x.get("estimated") for x in series[k])
            ax.bar(xs, d, w * 0.9, color=R.col(k), label=name + ("  (estimate*)" if est else ""),
                   hatch="///" if est else None, edgecolor="white", linewidth=0.6)
            for i in (1, 3, len(secs)):                         # verify, restore and the whole step
                ax.annotate(f"{d[i]:+.2f}", (xs[i], d[i]), xytext=(0, 3 if d[i] >= 0 else -3),
                            textcoords="offset points", ha="center", va="bottom" if d[i] >= 0 else "top",
                            fontsize=9, color=R.INK, rotation=90)
        ax.axhline(0, color=R.INK, linewidth=1.2)
        ax.axvspan(len(secs) - 0.5, len(secs) + 0.5, color=R.GRID, alpha=0.35, zorder=0)
        ax.set_xticks(range(len(secs) + 1), [n for n, _ in secs] + ["whole step"], fontsize=12)
        ax.tick_params(axis="y", labelsize=11, colors=R.MUTED)
        ax.tick_params(axis="x", colors=R.INK)
        ax.set_ylabel("ms per decode step vs SpecLA  (+ = slower)", fontsize=12, color=R.MUTED)
        ax.set_title(f"Batch {b}: where each strategy loses to SpecLA (SpecLA step = "
                     f"{1e3 * rb['decode_s'] / rb['steps']:.2f} ms)", fontsize=15, color=R.INK, loc="left", pad=14)
        ax.grid(axis="y", color=R.GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        for sp in ("left", "bottom"):
            ax.spines[sp].set_color(R.GRID)
        lo, hi = ax.get_ylim()
        ax.set_ylim(min(lo, -0.05 * hi) * 1.25, hi * 1.25)
        ax.legend(loc="upper left", frameon=False, fontsize=11, ncol=1)
        fig.text(0.01, 0.01, f"CUDA-graph engine, RTX 3090 (GPU 0), Qwen3.5-4B + DFlash-b16, GSM8K-256, mean of 3 runs. "
                 "Snapshot writes happen inside the verify kernel."
                 + (note or ("\noracle was measured ~6 h after the other strategies (run-to-run conditions differ by up "
                             "to ~1%)." if "oracle" in series else "")), fontsize=9, color=R.MUTED)
        fig.tight_layout(rect=(0, 0.07 if note else 0.05, 1, 1))
        for ext in ("png", "svg"):
            fig.savefig(os.path.join(outdir, f"delta_vs_specla_bs{b}{suffix}.{ext}"), bbox_inches="tight")
        plt.close(fig)


def modeal_al_hist(d, path, batches=(1, 8, 64)):
    """modeal: share of row-steps per acceptance length (al_hist summed over runs), coloured by the restore path the
    step takes relative to the dominant m (AL < m: replay from S_0, AL = m: snapshot copy only, AL > m: replay from m)."""
    cells = {b: json.load(open(p)) for b in batches
             if os.path.exists(p := os.path.join(ROOT, d, "cells", f"dflash_bs{b}.json"))}
    cells = {b: c for b, c in cells.items() if c.get("status") == "done"}
    if not cells:
        return
    paths = [("replay from S_0 (AL < m)", "#2a78d6"), ("snapshot copy only (AL = m)", "#eb6834"),
             ("replay from snapshot m (AL > m)", "#1baf7a")]       # validated categorical slots 1-3
    fig, axes = plt.subplots(1, len(cells), figsize=(4.6 * len(cells), 3.7), dpi=150, squeeze=False, sharey=True)
    for ax, (b, c) in zip(axes[0], cells.items()):
        H = [sum(r["al_hist"][a] for r in c["runs"]) for a in range(1, 17)]
        tot = sum(H)
        pct = [100 * h / tot for h in H]
        mc = {}
        for r in c["runs"]:
            for k, v in r["m_counts"].items():
                mc[int(k)] = mc.get(int(k), 0) + v
        m = max(mc, key=mc.get)
        grp = [0 if a < m else 1 if a == m else 2 for a in range(1, 17)]
        ax.bar(range(1, 17), pct, width=0.8, color=[paths[g][1] for g in grp], edgecolor="white", linewidth=1)
        for a in (m, 16):
            ax.annotate(f"{pct[a - 1]:.1f}%", (a, pct[a - 1]), xytext=(0, 3), textcoords="offset points",
                        ha="center", fontsize=7, color=R.INK)
        mean = sum(a * h for a, h in zip(range(1, 17), H)) / tot
        ax.axvline(mean, color=R.MUTED, linewidth=1, linestyle="--")
        tau = st.mean(r["tau"] for r in c["runs"])      # tau counts the last step after the 128-token cap
        ax.text(mean + 0.2, max(pct) * 0.97, f"mean {mean:.2f}\n(τ {tau:.2f} after the length cap)", fontsize=7,
                color=R.MUTED, va="top")
        share = [sum(p for p, g in zip(pct, grp) if g == i) for i in range(3)]
        R._style(ax, f"batch {b}", "% of row-steps" if b == next(iter(cells)) else "")
        ax.text(0, 1.02, f"m = {m} in {100 * mc[m] / sum(mc.values()):.0f}% of steps  ·  "
                         f"S_0 {share[0]:.1f}% / copy {share[1]:.1f}% / from m {share[2]:.1f}%",
                transform=ax.transAxes, fontsize=7, color=R.MUTED)
        ax.set_title(f"batch {b}", color=R.INK, fontsize=11, loc="left", pad=16)
        ax.set_xlabel("acceptance length (tokens committed in the step)", color=R.MUTED, fontsize=8)
        ax.set_xticks(range(1, 17))
        ax.grid(axis="x", visible=False)
    for name, col in paths:
        axes[0][0].bar([0], [0], color=col, label=name)
    axes[0][0].set_xlim(0.3, 16.7)
    fig.legend(loc="lower center", ncol=3, frameon=False, fontsize=8, bbox_to_anchor=(0.5, -0.04))
    fig.suptitle(f"modeal: acceptance-length distribution (GPU 0, {ENGINE} engine, 3 runs pooled)", color=R.INK,
                 fontsize=11, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def oracle_panels(data, path, batches=(1, 8, 64)):
    """Speedup over AR per batch size: SpecLA, oracle (true AL, one snapshot), modeal, fullstate as labelled
    horizontal bars (mean ± std of 3 runs), one panel per batch size; OOM cells are named."""
    keys = [k for k in ("specla_gpu0", "oracle", "modeal_gpu0", "fullstate") if k in data]
    names = {"specla_gpu0": "SpecLA (no snapshot)", "oracle": "oracle (true AL, 1 snapshot)",
             "modeal_gpu0": "modeal (mode AL, 1 snapshot)", "fullstate": "fullstate (16 snapshots)"}
    fig, axes = plt.subplots(1, len(batches), figsize=(4.4 * len(batches), 2.6), dpi=150, squeeze=False)
    for ax, b in zip(axes[0], batches):
        rows = {k: next((r for r in data[k] if r["batch"] == b), {}) for k in keys}
        vals = [r.get("speedup") for r in rows.values() if r.get("speedup") is not None]
        top = max(vals) if vals else 1
        for y, k in enumerate(keys):
            r = rows[k]
            if r.get("speedup") is None:
                ax.text(0.02 * top, y, " OOM", va="center", fontsize=7, color=R.MUTED)
                continue
            ax.barh(y, r["speedup"], height=0.6, color=R.col(k), xerr=r.get("speedup_std"),
                    error_kw=dict(ecolor=R.MUTED, lw=0.8, capsize=2))
            ax.text(r["speedup"] + (r.get("speedup_std") or 0) + 0.02 * top, y,
                    f"{r['speedup']:.2f}× ({r['throughput']:.0f})", va="center", fontsize=7, color=R.INK)
        ax.set_yticks(range(len(keys)), [names[k] for k in keys], fontsize=7)
        ax.invert_yaxis()
        R._style(ax, f"batch {b}", "")
        ax.set_xlim(0, top * 1.35)
        ax.set_xlabel("speedup over AR (×), tok/s in brackets", color=R.MUTED, fontsize=8)
        ax.grid(axis="x", color=R.GRID, linewidth=0.8)
        ax.grid(axis="y", visible=False)
        if b != batches[0]:
            ax.set_yticklabels([])
    fig.suptitle(f"Perfect AL prediction (oracle) vs SpecLA and snapshot strategies ({ENGINE} engine, GPU 0)",
                 color=R.INK, fontsize=10, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def eager_vs_graph(graph_data, graph_ars):
    """Same strategy, same GPU (0), eager vs CUDA-graph engine: throughput and speedup over that engine's AR."""
    eager_dirs = {"ar": "results_fullstate", "fullstate": "results_fullstate", "modeal_gpu0": "results_modeal_gpu0",
                  "specla_gpu0": "results_specla_gpu0", **{f"confk_{v}": f"results_confk_{v}" for v in CONFK_V}}
    lines = ["| 방식 | batch | eager tok/s | graph tok/s | graph/eager | eager speedup | graph speedup |",
             "|---|---|---|---|---|---|---|"]
    ar_e, _ = R.cell_rows(os.path.join(ROOT, "results_fullstate"), "fullstate")
    ar_e = {r["batch"]: r for r in ar_e if r["key"] == "ar"}
    ar_g = {r["batch"]: r for r in (graph_ars.get("0") or [])}
    for k in ["ar", "fullstate", "modeal_gpu0", "specla_gpu0"] + [f"confk_{v}" for v in CONFK_V]:
        d = os.path.join(ROOT, eager_dirs[k])
        if k == "ar":
            e_rows, g_rows = ar_e, ar_g
        else:
            if not os.path.isdir(os.path.join(d, "cells")) or k not in graph_data:
                continue
            rows, _ = R.cell_rows(d, k.replace("_gpu0", ""))
            e_rows = {r["batch"]: r for r in rows if r["key"] != "ar"}
            g_rows = {r["batch"]: r for r in graph_data[k]}
        for bs in R.BATCHES:
            e, g = e_rows.get(bs, {}), g_rows.get(bs, {})
            ce = json.load(open(os.path.join(d, "cells", f"{'ar' if k == 'ar' else 'dflash'}_bs{bs}.json"))) \
                if os.path.exists(os.path.join(d, "cells", f"{'ar' if k == 'ar' else 'dflash'}_bs{bs}.json")) else {}
            if (ce.get("cuda_visible_devices") or "0") != "0":
                continue                                   # eager cell measured on GPU 1: not comparable
            et, gt = e.get("throughput"), g.get("throughput")
            es = (et / ar_e[bs]["throughput"]) if et and ar_e.get(bs, {}).get("throughput") else None
            gs = (gt / ar_g[bs]["throughput"]) if gt and ar_g.get(bs, {}).get("throughput") else None
            f = lambda v, nd=1: "OOM" if v is None else f"{v:.{nd}f}"  # noqa: E731
            lines.append(f"| {R.lab('ar' if k == 'ar' else k)} | {bs} | {f(et)} | {f(gt)} | "
                         f"{f(gt / et if et and gt else None, 2)} | {f(es, 2)} | {f(gs, 2)} |")
    open(os.path.join(OUT, "eager_vs_graph.md"), "w").write("\n".join(lines) + "\n")


def main():
    os.makedirs(os.path.join(OUT, "plots"), exist_ok=True)
    for key, d in DIRS.items():
        if os.path.isdir(os.path.join(ROOT, d, "cells")):
            R.build(os.path.join(ROOT, d), ar_out=AR)
    data = {k: rows_for(k) for k in DIRS}
    data = {k: v for k, v in data.items() if v}
    ars = {g: ar_rows(g) for g in ("0", "1")}
    confk = {k: v for k, v in data.items() if k.startswith("confk_")}
    confkb = {k: v for k, v in data.items() if k.startswith("confkb_")}

    all_rows = [r for g in ars.values() if g for r in g] + [r for v in data.values() for r in v]
    R.write_csv(all_rows, os.path.join(OUT, "summary_all.csv"))
    open(os.path.join(OUT, "table_all.md"), "w").write(R.md_table(all_rows) + "\n")

    order = ["fullstate", "modeal_gpu0", "specla_gpu0", "oracle", "modeal", "specla"] + list(confk) + list(confkb)
    open(os.path.join(OUT, "table_speedup.md"), "w").write(
        speedup_table({k: data[k] for k in order if k in data}) + "\n")

    gpu0 = {"ar": ars["0"]} if ars["0"] else {}
    gpu0.update({k: v for k, v in data.items() if v and v[0].get("gpu") == "0"})
    open(os.path.join(OUT, "table_gpu0.md"), "w").write(R.md_table([r for v in gpu0.values() for r in v]) + "\n")

    cal = ["| 방식 | batch | GPU 0 tok/s | GPU 1 tok/s | GPU1/GPU0 |", "|---|---|---|---|---|"]
    for a, b, name in [("ar", "ar", "AR"), ("modeal_gpu0", "modeal", "mode-AL"), ("specla_gpu0", "specla", "SpecLA")]:
        ra = ars["0"] if a == "ar" else data.get(a)
        rb = ars["1"] if b == "ar" else data.get(b)
        if not ra or not rb:
            continue
        for bs in R.BATCHES:
            x = next((r for r in ra if r["batch"] == bs and r.get("throughput")), None)
            y = next((r for r in rb if r["batch"] == bs and r.get("throughput")), None)
            if x and y:
                cal.append(f"| {name} | {bs} | {x['throughput']:.1f} | {y['throughput']:.1f} | "
                           f"{y['throughput'] / x['throughput']:.3f} |")
    open(os.path.join(OUT, "gpu_calibration.md"), "w").write("\n".join(cal) + "\n")

    st_lines = ["| 방식 | GPU | batch | slot/시퀀스 | hit율 (복사만) | S_0에서 재계산 비율 | 스텝당 재계산 토큰 |",
                "|---|---|---|---|---|---|---|"]
    for k in list(confk) + list(confkb) + ["oracle", "modeal", "modeal_gpu0"]:
        if k not in data:
            continue
        gpu = data[k][0].get("gpu", "–")
        for bs, s in stats_rows(k):
            if "hit" in s:
                st_lines.append(f"| {R.lab(k)} | {gpu} | {bs} | {s['slots']} | {100 * s['hit']:.1f}% | "
                                f"{100 * s['s0']:.1f}% | {s['replay']:.2f} |")
            else:
                st_lines.append(f"| {R.lab(k)} | {gpu} | {bs} | 1 | – | – | m={s['m']} (스텝별 m 분포 {s['m_counts']}) |")
    open(os.path.join(OUT, "rollback_stats.md"), "w").write("\n".join(st_lines) + "\n")

    best, common, score = best_confk(confk) if confk else (None, [], {})
    best_b, common_b, score_b = best_confk(confkb) if confkb else (None, [], {})
    json.dump(dict(best=best, over_batches=common, mean_speedup=score, best_B=best_b, over_batches_B=common_b,
                   mean_speedup_B=score_b), open(os.path.join(OUT, "best_confk.json"), "w"), indent=1)

    # same-GPU (GPU 0) throughput + all-method speedup figures
    main_series = {"ar": ars["0"]} if ars["0"] else {}
    for k in ("fullstate", "modeal_gpu0", "specla_gpu0", "oracle"):
        if k in data:
            main_series[k] = data[k]
    for k in (best, best_b):
        if k:
            main_series[k] = data[k]
    R.plots(main_series, os.path.join(OUT, "plots"))
    for f in os.listdir(os.path.join(OUT, "plots")):
        if not f.startswith(("main_", "confk_", "restore_", "step_", "modeal_", "oracle_", "delta_")):
            os.replace(os.path.join(OUT, "plots", f), os.path.join(OUT, "plots", "main_" + f))
    if confk:
        refs = {k: data[k] for k in ("fullstate", "modeal_gpu0", "specla_gpu0") if k in data}
        confk_panels(confk, refs, os.path.join(OUT, "plots", "confk_speedup.png"), confkb)
    groups = {k: R.restore_by_al(os.path.join(ROOT, DIRS[k]), 1)
              for k in ("fullstate", "modeal_gpu0", "specla_gpu0", "oracle", best, best_b) if k in data}
    groups = {k: g for k, g in groups.items() if g}
    if groups:
        R.restore_plot(groups, 1, os.path.join(OUT, "plots", "restore_by_al_bs1.png"))
    bd = {k: v for k, v in gpu0.items() if k != "ar"}
    if ENGINE == "graph":           # one bar per strategy: the main three plus the best conf-K A / B variant
        bd = {k: bd[k] for k in ("fullstate", "modeal_gpu0", "specla_gpu0", "oracle", best, best_b) if k in bd}
    if bd:
        step_breakdown(bd, os.path.join(OUT, "plots", "step_breakdown_gpu0.png"))
    # per-section bars: one colour per strategy, so a single conf-K variant per method (GPU 0, same as the tables)
    sk = ("specla_gpu0", "oracle", "modeal_gpu0", "confk_thr0.3", "confkb_top1", "fullstate")
    sd = {k: gpu0[k] for k in sk if k in gpu0}
    if sd:
        step_sections(sd, os.path.join(OUT, "plots", "step_sections_gpu0.png"))
    dk = ("oracle", "modeal_gpu0", "confk_thr0.3", "confkb_top1", "fullstate")
    if "specla_gpu0" in gpu0:
        delta_vs_specla({k: gpu0[k] for k in dk if k in gpu0}, gpu0["specla_gpu0"],
                        os.path.join(OUT, "plots", "delta_vs_specla_gpu0.png"))
        delta_vs_specla_ppt({k: gpu0[k] for k in dk if k in gpu0}, gpu0["specla_gpu0"], os.path.join(OUT, "plots", "ppt"))
        if {"oracle", "modeal_gpu0"} <= set(gpu0):
            est = {k: gpu0[k] for k in dk if k in gpu0}
            est["oracle"] = oracle_estimate(gpu0["oracle"], gpu0["specla_gpu0"], gpu0["modeal_gpu0"])
            delta_vs_specla_ppt(est, gpu0["specla_gpu0"], os.path.join(OUT, "plots", "ppt"), suffix="_oracle_est",
                                note="\n* oracle = estimate (measured ~6 h later; identical work ran 1.1-2.0% slower then).\n"
                                     f"  draft = SpecLA's (same kernels); verify = modeal's - {SLOTMAP_MS} ms slot-map build "
                                     "(modeal builds it inside verify); other sections = measured / session slowdown.")
    modeal_al_hist(DIRS["modeal_gpu0"], os.path.join(OUT, "plots", "modeal_al_hist.png"))
    if "oracle" in data:
        oracle_panels(data, os.path.join(OUT, "plots", "oracle_speedup.png"))
    if ENGINE == "graph":
        eager_vs_graph(data, ars)
    for f in ("table_speedup.md", "gpu_calibration.md", "rollback_stats.md") + (("eager_vs_graph.md",) if ENGINE == "graph" else ()):
        print(f"== {f}\n" + open(os.path.join(OUT, f)).read())
    print("best conf-K:", best, {k: round(v, 3) for k, v in score.items()})
    print("best conf-K B:", best_b, {k: round(v, 3) for k, v in score_b.items()})


if __name__ == "__main__":
    main()
