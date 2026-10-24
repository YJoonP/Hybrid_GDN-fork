"""summary.csv, plots/ and the data sections of report.md from results_<rollback>/ (cells/*.json + raw/*.jsonl).

Measured-only numbers are kept separate from derived ones: the GDN snapshot-write overhead inside the fused verify
kernel cannot be timed in-run without perturbing it, so it is reported as the per-step calibration measurement and,
where a total is needed, as calibration x steps (labelled "derived")."""
import csv
import glob
import json
import os
import statistics as st
from collections import defaultdict

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

BATCHES = [1, 8, 64, 128]
COLORS = {"ar": "#2a78d6", "fullstate": "#eb6834", "modeal": "#1baf7a", "specla": "#eda100",
          "confk": "#e87ba4", "confkb": "#008300", "oracle": "#6250d6",
          "oracle_copy": "#e34948"}                                      # validated categorical slots 1-8
LABELS = {"ar": "AR", "fullstate": "DFlash-b16 (full state)", "modeal": "DFlash-b16 (mode-AL snapshot + recompute)",
          "specla": "DFlash-b16 (SpecLA, recompute)", "oracle": "DFlash-b16 (oracle AL, 1 snapshot)",
          "oracle_copy": "DFlash-b16 (oracle AL, copy-only restore)"}


def col(k):
    return COLORS.get(k, COLORS.get(k.split("_")[0], MUTED))


def lab(k):
    if k in LABELS:
        return LABELS[k]
    if k.startswith(("confk_", "confkb_")):
        pre, v = k.split("_", 1)
        sel = 'top-' + v[3:] if v.startswith('top') else 'θ=' + v[3:]
        return f"DFlash-b16 (conf-K, {sel})" if pre == "confk" else f"DFlash-b16 (conf-K B head, {sel})"
    return k
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"


def _ms(xs):
    xs = [x for x in xs if x is not None]
    if not xs:
        return None, None
    return st.mean(xs), (st.stdev(xs) if len(xs) > 1 else 0.0)


def _raw(out, name, run):
    p = os.path.join(out, "raw", f"{name}_run{run}.jsonl")
    return [json.loads(l) for l in open(p)] if os.path.exists(p) else None


def _match(a, b):
    """fraction of identical outputs, mean common-prefix fraction"""
    if not a or not b:
        return None, None
    da = {x["order"]: x["output"] for x in a}
    same = pref = 0
    for x in b:
        y = da[x["order"]]
        same += x["output"] == y
        k = next((j for j in range(min(len(y), len(x["output"]))) if y[j] != x["output"][j]), min(len(y), len(x["output"])))
        pref += k / max(len(y), 1)
    return same / len(b), pref / len(b)


def required_snapshot_gb(c, key, bs):
    """Allocated snapshot size, or (OOM before allocation) the size this strategy would need."""
    if c.get("snapshot_allocated"):
        return c["snapshot_allocated"]["total_bytes"] / 1e9
    f = c.get("snapshot_formula")          # per-state sizes (written for every DFlash cell by earlier code)
    if not f:
        return None
    per_state = f["rec_state_bytes"] + f["conv_state_bytes"]
    key = key.replace("_gpu0", "")          # same-GPU replication dirs
    if key == "fullstate":
        slots, per = f["positions"], per_state
    elif key == "modeal":
        slots, per = 1, per_state
    elif key == "specla":
        return 0.0
    elif key in ("oracle", "oracle_copy"):  # one recurrent-state slot (conf-K storage)
        slots, per = 1, f["rec_state_bytes"]
    else:                                   # confk: recurrent-state slots only
        v = key.split("_", 1)[1]
        slots, per = (int(v[3:]) + 1 if v.startswith("top") else 2), f["rec_state_bytes"]
    return f["layers"] * slots * per * bs / 1e9


def cell_rows(out, method_key):
    """One summary row per cell of a results dir. method_key: 'fullstate' / 'specla' labels the DFlash rows."""
    cells = {os.path.basename(p)[:-5]: json.load(open(p)) for p in glob.glob(os.path.join(out, "cells", "*.json"))}
    rows = []
    for method in ("ar", "dflash"):
        for bs in BATCHES:
            c = cells.get(f"{method}_bs{bs}")
            key = "ar" if method == "ar" else method_key
            row = dict(method=lab(key), key=key, batch=bs)
            if c is None:
                row["status"] = "not run"
                rows.append(row)
                continue
            row["status"] = c["status"] + (" (reused)" if c.get("reused_from") else "")
            runs = c.get("runs", [])
            row["n_runs"] = len(runs)
            row["peak_mem_gb"] = c.get("peak_mem_gb")
            row["snapshot_allocated"] = bool(c.get("snapshot_allocated"))
            row["snapshot_gb"] = None if method == "ar" else required_snapshot_gb(c, key, bs)
            if c["status"] == "oom":
                row["oom_stage"] = c.get("stage")
            if runs:
                for k, f in [("throughput", lambda r: r["throughput"]), ("tau", lambda r: r["tau"]),
                             ("tpot_ms", lambda r: r["tpot_ms"]), ("decode_s", lambda r: r["decode_s"]),
                             ("prefill_s", lambda r: r["prefill_s"])]:
                    row[k], row[k + "_std"] = _ms([f(r) for r in runs])
                for sec in ("draft", "verify", "save", "restore", "kv_truncate", "accept", "remove", "host_gap"):
                    row["t_" + sec], row["t_" + sec + "_std"] = _ms([r["sections"].get(sec, 0.0) for r in runs])
                row["t_gdn_kernel"] = st.mean(r["gdn_verify_kernel_s"] for r in runs)
                row["steps"] = st.mean(r["steps"] for r in runs)
                cal = c.get("store_calibration")
                if cal:
                    row["gdn_store_ms_per_step"] = cal["step_ms_save_overhead"]
                    row["t_gdn_store_derived"] = cal["step_ms_save_overhead"] * row["steps"] / 1e3
            rows.append(row)
    return rows, cells


def gpu_of(cell):
    return cell.get("cuda_visible_devices") or "0"


def add_relative(rows, out, ar_out):
    """speedup vs AR of the same batch on the same GPU (paired by run index), output match rate vs that AR's run 1.
    ar_out: results dir holding the AR cells, or {gpu index: dir} to pick the AR measured on the cell's GPU."""
    ar_dirs = ar_out if isinstance(ar_out, dict) else None
    for r in rows:
        bs = r["batch"]
        if r.get("throughput") is None:
            continue
        cell = json.load(open(os.path.join(out, "cells", f"{'ar' if r['key'] == 'ar' else 'dflash'}_bs{bs}.json")))
        src = ar_dirs.get(gpu_of(cell)) if ar_dirs else ar_out
        r["gpu"] = gpu_of(cell)
        p = os.path.join(src, "cells", f"ar_bs{bs}.json") if src else None
        ar = json.load(open(p)) if p and os.path.exists(p) else None
        if not ar or ar.get("status") != "done":
            continue
        ar_src = src
        sp = [x["throughput"] / y["throughput"] for x, y in zip(cell["runs"], ar["runs"])]
        r["speedup"], r["speedup_std"] = _ms(sp)
        ref = _raw(ar_src, "ar_bs1" if r["key"] == "ar" else f"ar_bs{bs}", 1)
        mine = _raw(out, f"{'ar' if r['key'] == 'ar' else 'dflash'}_bs{bs}", 1)
        r["match_ref"] = "AR bs=1 run1" if r["key"] == "ar" else f"AR bs={bs} run1"
        r["match_identical"], r["match_prefix"] = _match(ref, mine)
        if r["key"] != "ar":
            dec = r["decode_s"]
            meas = (r["t_save"] + r["t_restore"]) / dec
            r["rollback_share_measured"] = meas
            if r.get("t_gdn_store_derived") is not None:
                r["rollback_share_with_gdn_store"] = meas + r["t_gdn_store_derived"] / dec
    return rows


def restore_by_al(out, bs):
    """per-step restore time grouped by the step's acceptance length (bs=1: exact AL; bs>1: rounded mean AL)."""
    groups = defaultdict(list)
    for p in sorted(glob.glob(os.path.join(out, "raw", f"dflash_bs{bs}_run*_steps.jsonl"))):
        for l in open(p):
            s = json.loads(l)
            if s.get("produced"):
                al = s["produced"][0] if bs == 1 else round(st.mean(s["produced"]))
                groups[al].append(s["restore"] * 1e3)
    return dict(sorted(groups.items()))


COLS = [("method", "method"), ("batch", "batch"), ("status", "status"), ("n_runs", "runs"),
        ("throughput", "decode tok/s"), ("throughput_std", "±"), ("speedup", "speedup"), ("speedup_std", "±"),
        ("tau", "tau"), ("tau_std", "±"), ("tpot_ms", "TPOT ms"), ("tpot_ms_std", "±"),
        ("t_draft", "draft s"), ("t_verify", "verify s"), ("t_save", "save s (conv, measured)"),
        ("gdn_store_ms_per_step", "GDN store ms/step (calib.)"), ("t_gdn_store_derived", "GDN store s (derived)"),
        ("t_restore", "restore s"), ("t_kv_truncate", "KV truncate s"), ("t_accept", "accept/write s"),
        ("t_remove", "slot remove s"), ("decode_s", "decode s"), ("prefill_s", "prefill s (excluded)"),
        ("rollback_share_measured", "rollback share (measured)"),
        ("rollback_share_with_gdn_store", "rollback share (+GDN store, derived)"),
        ("peak_mem_gb", "peak mem GB"), ("snapshot_gb", "snapshot GB"), ("match_identical", "identical vs AR"),
        ("match_prefix", "mean prefix match"), ("match_ref", "match ref")]


def write_csv(rows, path):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([c[0] for c in COLS])
        for r in rows:
            w.writerow(["" if r.get(k) is None else (f"{r[k]:.6g}" if isinstance(r[k], float) else r[k]) for k, _ in COLS])


def _fmt(v, nd=1, pct=False):
    if v is None:
        return "–"
    if pct:
        return f"{100 * v:.1f}%"
    return f"{v:.{nd}f}" if isinstance(v, float) else str(v)


def md_table(rows):
    h = ["방식", "GPU", "batch", "상태", "decode tok/s", "speedup", "τ", "TPOT ms", "draft s", "verify s", "save s (conv)",
         "GDN 저장 ms/step (보정)", "restore s", "KV trunc s", "peak GB", "snapshot GB", "AR 대비 동일 출력"]
    lines = ["| " + " | ".join(h) + " |", "|" + "---|" * len(h)]
    for r in rows:
        pm = lambda a, b, nd=1: "–" if r.get(a) is None else f"{r[a]:.{nd}f} ± {r[b]:.{nd}f}"  # noqa: E731
        lines.append("| " + " | ".join([
            r["method"], str(r.get("gpu", "–")), str(r["batch"]), r["status"], pm("throughput", "throughput_std"),
            pm("speedup", "speedup_std", 2), pm("tau", "tau_std", 2), pm("tpot_ms", "tpot_ms_std"),
            _fmt(r.get("t_draft")), _fmt(r.get("t_verify")), _fmt(r.get("t_save"), 2),
            _fmt(r.get("gdn_store_ms_per_step"), 2), _fmt(r.get("t_restore"), 2), _fmt(r.get("t_kv_truncate"), 3),
            _fmt(r.get("peak_mem_gb")), _fmt(r.get("snapshot_gb"), 2) + ("" if r.get("snapshot_allocated", True)
                                                                         or r.get("snapshot_gb") is None else " (필요량)"),
            _fmt(r.get("match_identical"), pct=True)]) + " |")
    return "\n".join(lines)


def _style(ax, title, ylabel):
    ax.set_title(title, color=INK, fontsize=11, loc="left")
    ax.set_ylabel(ylabel, color=MUTED, fontsize=9)
    ax.set_xlabel("batch size (concurrent sequences)", color=MUTED, fontsize=9)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=8)


def plots(series, outdir):
    """series: {key: rows}. Bars per batch (categorical x), error bars = std over runs; OOM cells annotated."""
    os.makedirs(outdir, exist_ok=True)
    keys = list(series)
    x = list(range(len(BATCHES)))
    w = 0.8 / len(keys)

    def bar_plot(fname, title, ylabel, field, std=None, pct=False, skip_ar=False):
        fig, ax = plt.subplots(figsize=(7, 3.6), dpi=150)
        ks = [k for k in keys if not (skip_ar and k == "ar")]
        for j, k in enumerate(ks):
            rows = {r["batch"]: r for r in series[k]}
            labelled = False
            for i, bs in enumerate(BATCHES):
                r = rows.get(bs, {})
                xpos = i + (j - (len(ks) - 1) / 2) * w
                v = r.get(field)
                if v is None:
                    txt = "OOM" if str(r.get("status", "")).startswith("oom") else "n/a"
                    ax.text(xpos, 0, txt, ha="center", va="bottom", fontsize=7, color=MUTED, rotation=90)
                    continue
                v = 100 * v if pct else v
                e = (100 * r[std] if pct else r[std]) if std and r.get(std) is not None else None
                ax.bar(xpos, v, w * 0.92, color=col(k), label=None if labelled else lab(k),
                       yerr=e, error_kw=dict(ecolor=MUTED, lw=0.8, capsize=2))
                labelled = True
                vert = len(ks) >= 4
                ax.annotate(f"{v:.0f}" if v >= 100 else f"{v:.1f}" if v >= 10 else f"{v:.2f}", (xpos, v + (e or 0)),
                            xytext=(0, 3), textcoords="offset points", ha="center", va="bottom",
                            fontsize=6 if vert else 7, color=INK, rotation=90 if vert else 0)
        ax.set_xticks(x, [str(b) for b in BATCHES])
        _style(ax, title, ylabel)
        lo, hi = ax.get_ylim()
        ax.set_ylim(lo, hi * 1.12)                                   # headroom for the value labels
        h, l = ax.get_legend_handles_labels()
        by = dict(zip(l, h))
        if by:
            fig.legend(by.values(), by.keys(), frameon=False, fontsize=8, loc="lower center",
                       ncol=2 if len(by) > 2 else len(by), bbox_to_anchor=(0.5, -0.02))
        fig.tight_layout(rect=(0, 0.06 * ((len(by) + 1) // 2), 1, 1))
        fig.savefig(os.path.join(outdir, fname), bbox_inches="tight")
        plt.close(fig)

    bar_plot("throughput.png", "Decode throughput (mean ± std, 3 runs)", "tokens / s", "throughput", "throughput_std")
    bar_plot("speedup.png", "Speedup over AR at the same batch size", "× AR", "speedup", "speedup_std", skip_ar=True)
    bar_plot("tau.png", "Mean acceptance length τ (tokens per verify, incl. bonus)", "tokens", "tau", "tau_std",
             skip_ar=True)
    bar_plot("rollback_share.png", "Rollback time share of decode time (save + restore, measured)", "% of decode",
             "rollback_share_measured", pct=True, skip_ar=True)
    bar_plot("rollback_share_with_gdn_store.png", "Rollback share incl. GDN snapshot writes (calibration-derived)",
             "% of decode", "rollback_share_with_gdn_store", pct=True, skip_ar=True)
    bar_plot("peak_memory.png", "Peak GPU memory (torch max_memory_allocated)", "GB", "peak_mem_gb")
    bar_plot("snapshot_memory.png", "Snapshot buffer size (allocated, or required when OOM)", "GB", "snapshot_gb",
             skip_ar=True)


def restore_plot(groups_by_key, bs, path):
    fig, ax = plt.subplots(figsize=(7, 3.4), dpi=150)
    ks = list(groups_by_key)
    for j, k in enumerate(ks):
        g = groups_by_key[k]
        if not g:
            continue
        als = list(g)
        pct = {a: np.percentile(g[a], [10, 50, 90]) for a in als}    # median: restore times are heavy-tailed
        means = [pct[a][1] for a in als]
        lo = [pct[a][1] - pct[a][0] for a in als]
        hi = [pct[a][2] - pct[a][1] for a in als]
        xs = [a + (j - (len(ks) - 1) / 2) * 0.3 for a in als]
        ax.errorbar(xs, means, yerr=[lo, hi], fmt="o", ms=4, color=col(k), ecolor=col(k), elinewidth=0.8,
                    capsize=2, label=lab(k))
    _style(ax, f"State restore time by acceptance length (batch {bs}; median, P10–P90)", "ms per step")
    ax.set_xlabel("acceptance length (tokens committed in the step)" + ("" if bs == 1 else ", rounded batch mean"),
                  color=MUTED, fontsize=9)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def build(out, ar_out=None):
    """Per-dir artefacts: summary.csv, plots/, report_data.md (tables used by report.md)."""
    key = os.path.basename(os.path.normpath(out)).replace("results_", "")
    rows, cells = cell_rows(out, key)
    rows = add_relative(rows, out, ar_out or out)
    write_csv(rows, os.path.join(out, "summary.csv"))
    plots({"ar": [r for r in rows if r["key"] == "ar"], key: [r for r in rows if r["key"] == key]},
          os.path.join(out, "plots"))
    for bs in (1, 8, 64, 128):
        g = restore_by_al(out, bs)
        if g:
            restore_plot({key: g}, bs, os.path.join(out, "plots", f"restore_by_al_bs{bs}.png"))
    with open(os.path.join(out, "report_data.md"), "w") as f:
        f.write(md_table(rows) + "\n")
    return rows


def build_compare(dirs, outdir, ar_out):
    """Side-by-side plots / table for several rollback dirs that share the AR cells of ar_out."""
    os.makedirs(outdir, exist_ok=True)
    series, all_rows = {}, []
    for d in dirs:
        key = os.path.basename(os.path.normpath(d)).replace("results_", "")
        rows, _ = cell_rows(d, key)
        rows = add_relative(rows, d, ar_out)
        series.setdefault("ar", [r for r in rows if r["key"] == "ar"])
        series[key] = [r for r in rows if r["key"] == key]
        all_rows += series[key]
    plots(series, os.path.join(outdir, "plots"))
    for bs in (1, 8, 64, 128):
        gs = {os.path.basename(os.path.normpath(d)).replace("results_", ""): restore_by_al(d, bs) for d in dirs}
        if any(gs.values()):
            restore_plot(gs, bs, os.path.join(outdir, "plots", f"restore_by_al_bs{bs}.png"))
    rows = series["ar"] + all_rows
    write_csv(rows, os.path.join(outdir, "summary.csv"))
    with open(os.path.join(outdir, "report_data.md"), "w") as f:
        f.write(md_table(rows) + "\n")
    return rows
