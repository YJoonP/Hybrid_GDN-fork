"""Slide figures (16:9, 300-dpi PNG + SVG with text as paths) for the CUDA-graph results, Korean labels.

  delta_bs{1,8,64}.*   SpecLA 대비 단계별 시간 차이: draft / verify / restore / 스텝 전체
  tps_bs{1,8,64}.*     처리량(TPS), AR 대비 배수
  tps_all.*            batch 1 / 8 / 64 처리량을 한 장에 (패널마다 y축 따로)

The oracle bars are the session-corrected estimate (analyze_all.oracle_estimate, hatched); the same figures with the
measured oracle go to measured/. Output: results_compare_graph/plots/ppt/.
"""
import os
import sys

sys.argv = [sys.argv[0], "--engine", "graph"]
import analyze_all as A  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

R = A.R
plt.rcParams["font.family"] = ["Noto Sans CJK KR", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["svg.fonttype"] = "path"

OUT = os.path.join(A.OUT, "plots", "ppt")
METHODS = [("specla_gpu0", "SpecLA"), ("oracle", "Oracle"), ("modeal_gpu0", "AL 빈도 기준 저장 (modeal)"),
           ("confk_thr0.3", "softmax 기준 저장 (conf-K A)"), ("confkb_top1", "confidence head (conf-K B)"),
           ("fullstate", "full state")]
COLOR = {"ar": R.COLORS["ar"], "specla_gpu0": R.COLORS["specla"], "oracle": R.COLORS["oracle"],
         "modeal_gpu0": R.COLORS["modeal"], "confk_thr0.3": R.COLORS["confk"], "confkb_top1": R.COLORS["confkb"],
         "fullstate": R.COLORS["fullstate"]}
BASE_NOTE = ("CUDA graph 엔진, RTX 3090 (GPU 0), Qwen3.5-4B + DFlash-b16, GSM8K 256문항, 3회 평균. "
             "conf-K A = θ 0.3, conf-K B = top-1 (각 방식에서 가장 빠른 변형).")
EST_NOTE = ("* Oracle = 추정치: 다른 방식보다 약 6시간 뒤에 측정되어 같은 작업이 1.1~2.0% 느렸던 시간대를 보정. "
            "draft = SpecLA와 같음, verify = modeal − 0.16 ms,\n  나머지 = 실측 ÷ 그 시간대 감속률.   "
            "† modeal의 draft는 SpecLA와 같은 커널이라 같게 맞춤 (측정 차이 ≤ 0.10 ms).")
MEAS_NOTE = "Oracle은 다른 방식보다 약 6시간 뒤에 측정 (측정 시점에 따라 같은 작업도 최대 약 2% 차이)."


def style(ax):
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color(R.GRID)
    ax.grid(axis="y", color=R.GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(axis="y", labelsize=11, colors=R.MUTED)


def save(fig, outdir, name):
    os.makedirs(outdir, exist_ok=True)
    for ext in ("png", "svg"):
        fig.savefig(os.path.join(outdir, f"{name}.{ext}"), bbox_inches="tight", dpi=300)
    plt.close(fig)


def row(rows, b):
    return next((r for r in rows if r["batch"] == b), {})


def delta(data, outdir, note):
    secs = [("draft", ("draft",)), ("verify\n(State 저장 포함)", ("verify",)), ("restore", ("restore",))]
    keys = [k for k, _ in METHODS[1:] if k in data]
    w = 0.8 / len(keys)
    per = lambda r, f: 1e3 * sum(r.get("t_" + x) or 0 for x in f) / r["steps"]  # noqa: E731
    for b in (1, 8, 64):
        rb = row(data["specla_gpu0"], b)
        fig, ax = plt.subplots(figsize=(13.33, 7.5))
        for j, k in enumerate(keys):
            r = row(data[k], b)
            name = dict(METHODS)[k]
            xs = [i + (j - (len(keys) - 1) / 2) * w for i in range(len(secs) + 1)]
            est = r.get("estimated")
            if not r.get("steps"):
                ax.bar([0], [0], color=COLOR[k], label=name + "  — 메모리 부족 (OOM)")
                ax.text(xs[-1], 0, " OOM", rotation=90, ha="center", va="bottom", fontsize=11, color=R.MUTED)
                continue
            d = [per(r, f) - per(rb, f) for _, f in secs]
            d.append(1e3 * (r["decode_s"] / r["steps"] - rb["decode_s"] / rb["steps"]))
            ax.bar(xs, d, w * 0.9, color=COLOR[k], label=name + ("  (추정)" if est else ""),
                   hatch="///" if est else None, edgecolor="white", linewidth=0.6)
            for i, v in enumerate(d):
                if abs(v) >= 0.005:
                    ax.annotate(f"{v:+.2f}", (xs[i], v), xytext=(0, 3 if v >= 0 else -3), textcoords="offset points",
                                ha="center", va="bottom" if v >= 0 else "top", fontsize=10, color=R.INK, rotation=90)
        ax.axhline(0, color=R.INK, linewidth=1.2)
        ax.axvspan(len(secs) - 0.5, len(secs) + 0.5, color=R.GRID, alpha=0.35, zorder=0)
        ax.set_xticks(range(len(secs) + 1), [n for n, _ in secs] + ["스텝 전체"], fontsize=14)
        ax.set_ylabel("SpecLA 대비 스텝당 시간 차이 (ms, + = 느림)", fontsize=13, color=R.MUTED)
        ax.set_title(f"Batch {b}: SpecLA 대비 단계별 시간 차이  (SpecLA 스텝 = {1e3 * rb['decode_s'] / rb['steps']:.2f} ms)",
                     fontsize=17, color=R.INK, loc="left", pad=14)
        style(ax)
        lo, hi = ax.get_ylim()
        ax.set_ylim(min(lo, -0.05 * hi) * 1.25, hi * 1.25)
        ax.legend(loc="upper left", frameon=False, fontsize=12)
        fig.tight_layout()
        save(fig, outdir, f"delta_bs{b}")


def tps_panel(ax, data, ar, b, big):
    keys = ["ar"] + [k for k, _ in METHODS if k in data]
    names = {"ar": "AR", **dict(METHODS)}
    ra = row(ar, b)
    for x, k in enumerate(keys):
        r = ra if k == "ar" else row(data[k], b)
        if not r.get("throughput"):
            ax.text(x, 0, " OOM", rotation=90, ha="center", va="bottom", fontsize=11 if big else 9, color=R.MUTED)
            continue
        est = r.get("estimated")
        v = r["throughput"]
        ax.bar(x, v, 0.72, color=COLOR[k], hatch="///" if est else None, edgecolor="white", linewidth=0.6,
               yerr=None if est else r.get("throughput_std"), error_kw=dict(ecolor=R.MUTED, lw=0.8, capsize=3))
        sp = v / ra["throughput"] if ra.get("throughput") else None
        lab = f"{v:.0f}" + (f"\n({sp:.2f}×)" if sp and k != "ar" else "")
        ax.annotate(lab, (x, v), xytext=(0, 4), textcoords="offset points", ha="center", va="bottom",
                    fontsize=11 if big else 8.5, color=R.INK)
    ax.set_xticks(range(len(keys)), [names[k] for k in keys], rotation=30, ha="right", fontsize=12 if big else 9)
    ax.set_ylim(0, ax.get_ylim()[1] * 1.15)
    style(ax)


def tps(data, ar, outdir, note):
    for b in (1, 8, 64):
        fig, ax = plt.subplots(figsize=(13.33, 7.5))
        tps_panel(ax, data, ar, b, True)
        ax.set_ylabel("처리량 (tokens / s)", fontsize=13, color=R.MUTED)
        ax.set_title(f"Batch {b}: decode 처리량 (TPS), 괄호는 AR 대비 배수", fontsize=17, color=R.INK, loc="left", pad=14)
        fig.tight_layout()
        save(fig, outdir, f"tps_bs{b}")
    fig, axes = plt.subplots(1, 3, figsize=(13.33, 7.5))
    for ax, b in zip(axes, (1, 8, 64)):
        tps_panel(ax, data, ar, b, False)
        ax.set_title(f"Batch {b}", fontsize=14, color=R.INK, loc="left")
    axes[0].set_ylabel("처리량 (tokens / s)", fontsize=12, color=R.MUTED)
    fig.suptitle("decode 처리량 (TPS), 괄호는 같은 batch AR 대비 배수  (패널마다 y축 범위가 다름)", fontsize=15,
                 color=R.INK, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    save(fig, outdir, "tps_all")


def tps_grouped(data, ar, outdir, note, batches=(1, 8, 64, 128)):
    """One chart: batch size on x, one bar per strategy (AR first), value labels; OOM cells named."""
    keys = ["ar"] + [k for k, _ in METHODS if k in data]
    names = {"ar": "AR", **dict(METHODS)}
    w = 0.84 / len(keys)
    fig, ax = plt.subplots(figsize=(13.33, 7.5))
    for j, k in enumerate(keys):
        rows = ar if k == "ar" else data[k]
        est = any(r.get("estimated") for r in rows)
        adj = any(r.get("draft_equalized") for r in rows)
        labelled = False
        for i, b in enumerate(batches):
            x = i + (j - (len(keys) - 1) / 2) * w
            r = row(rows, b)
            if not r.get("throughput"):
                ax.text(x, 0, "OOM", rotation=90, ha="center", va="bottom", fontsize=8, color=R.MUTED)
                continue
            v = r["throughput"]
            ax.bar(x, v, w * 0.92, color=COLOR[k], hatch="///" if r.get("estimated") else None, edgecolor="white",
                   linewidth=0.5, label=None if labelled else names[k] + ("  (추정)" if est else ""),
                   yerr=None if r.get("estimated") else r.get("throughput_std"),
                   error_kw=dict(ecolor=R.MUTED, lw=0.7, capsize=1.5))
            labelled = True
            ax.annotate(f"{v:.0f}", (x, v), xytext=(0, 3), textcoords="offset points", ha="center", va="bottom",
                        fontsize=8.5, color=R.INK, rotation=90)
    ax.set_xticks(range(len(batches)), [f"batch {b}" for b in batches], fontsize=14)
    ax.set_xlim(-0.55, len(batches) - 0.45)
    ax.set_ylabel("처리량 (tokens / s)", fontsize=13, color=R.MUTED)
    ax.set_title("decode 처리량 (TPS), batch별", fontsize=17, color=R.INK, loc="left", pad=14)
    style(ax)
    ax.set_ylim(0, ax.get_ylim()[1] * 1.12)
    ax.legend(loc="upper left", frameon=False, fontsize=11.5, ncol=2)
    fig.tight_layout()
    save(fig, outdir, "tps_by_batch")


def main():
    data = {k: A.rows_for(k) for k, _ in METHODS}
    ar = A.ar_rows("0")
    est = dict(data, oracle=A.oracle_estimate(data["oracle"], data["specla_gpu0"], data["modeal_gpu0"]))
    for r in est["oracle"]:                      # TPS of the estimate relative to the same AR
        if r.get("estimated"):
            r["speedup"] = r["throughput"] / row(ar, r["batch"])["throughput"]
    md = []                                      # modeal: same draft kernels as SpecLA -> same draft time
    for r in data["modeal_gpu0"]:
        sp = row(data["specla_gpu0"], r["batch"])
        if r.get("steps") and sp.get("steps"):
            e = dict(r, draft_equalized=True, t_draft=sp["t_draft"])
            e["decode_s"] = r["decode_s"] + sp["t_draft"] - r["t_draft"]
            e["throughput"] = r["throughput"] * r["decode_s"] / e["decode_s"]
            md.append(e)
        else:
            md.append(r)
    est["modeal_gpu0"] = md
    delta(est, OUT, EST_NOTE)
    tps(est, ar, OUT, EST_NOTE)
    tps_grouped(est, ar, OUT, EST_NOTE)
    delta(data, os.path.join(OUT, "measured"), MEAS_NOTE)
    tps(data, ar, os.path.join(OUT, "measured"), MEAS_NOTE)
    tps_grouped(data, ar, os.path.join(OUT, "measured"), MEAS_NOTE)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
