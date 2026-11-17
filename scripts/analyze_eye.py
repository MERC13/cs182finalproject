"""Aligned vs rotated covariance with an axis-preserving (identity) read-in vs the standard Gaussian read-in.

  python scripts/analyze_eye.py
"""
import glob
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from icl.analysis import load  # noqa: E402
from analyze_confirm import run_metrics  # noqa: E402


def row(path):
    r = load(path)
    m = run_metrics(r)
    ax = [int(t) if np.isfinite(t) else None for t in m["T_axis"]]
    return dict(run=os.path.basename(path), steps=int(r.steps[-1]), T_conv=m["T_conv"], T_weak=m["T_weak"],
                axes=ax, err32=m["err32_stable"], pcr=m["pcr_over_ridge"])


def at_step(path, step):
    """err@32 and axis-probe state at a common step (to compare runs of different length)."""
    r = load(path)
    i = int(np.searchsorted(r.steps, step))
    i = min(i, len(r.steps) - 1)
    return float(r.err_last[i])


if __name__ == "__main__":
    groups = {"identity read-in (this test)": sorted(glob.glob("runs/why/*_eye_*.json")),
              "Gaussian read-in (confirmation, seeds 10-12)": sorted(glob.glob("runs/confirm/*_1i2-*_s1[0-2].json"))}
    for g, fs in groups.items():
        print(f"\n== {g}")
        for f in fs:
            m = row(f)
            fmt = lambda v: "never" if v is None or (isinstance(v, float) and not np.isfinite(v)) else f"{v:,.0f}"
            print(f"  {m['run']:34s} steps {m['steps']:5d}  converged {fmt(m['T_conv']):>6s}  weakest axis {fmt(m['T_weak']):>6s}"
                  f"  err@32 {m['err32']:.4f}  PCR/ridge {m['pcr']:.2f}  axes {m['axes']}")
