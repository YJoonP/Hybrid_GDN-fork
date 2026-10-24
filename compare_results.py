"""Side-by-side AR / full state / SpecLA tables and plots -> results_compare/ (AR cells from results_fullstate)."""
import os
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
from dflash_cb import report  # noqa: E402

dirs = [os.path.join(ROOT, d) for d in ("results_fullstate", "results_modeal", "results_specla")]
report.build_compare(dirs, os.path.join(ROOT, "results_compare"), ar_out=dirs[0])
print(open(os.path.join(ROOT, "results_compare", "report_data.md")).read())
