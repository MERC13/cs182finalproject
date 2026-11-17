"""Aggregate the confirmation runs (fresh seeds, held-out prompts) across seeds.

  python scripts/analyze_confirm.py runs/confirm figures/confirm
"""
from __future__ import annotations

import glob
import json
import os
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from icl.analysis import load  # noqa: E402
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from analyze_grid import dist_shape  # noqa: E402

CONDS = ["iso", "1/i2-ali", "1/i2-rot"]
LEVELS = [0.03, 0.01, 0.005]  # in-distribution query error levels for the matched-error H3 test


def run_metrics(stable, decay=None):
    r = stable
    exc = r.excess_mean()
    plateau = 1.0 - r.ref_err["ols"][1:].mean()
    out = dict(seed=r.cfg["seed"], lr=r.lr, beta2=r.cfg.get("beta2", 0.999),
               T_escape=r.first_below(r.err_mean, 0.9), T_half=r.first_below(exc, 0.5 * plateau),
               T_conv=r.first_below(exc, 0.05), err32_stable=float(r.err_last[-1]), errmean_stable=float(r.err_mean[-1]))
    T_ax = r.axis_learn_steps(0.5)
    out["T_axis"] = T_ax.tolist()
    out["T_weak"] = float(T_ax[-1])
    out["axis_ratio"] = float(T_ax[-1] / T_ax[0]) if np.isfinite(T_ax).all() else np.nan
    out["pcr_over_ridge"], _ = dist_shape(r)
    out["q1_stable"] = float(r.qshift[-1][-1])
    out["gain_weak_stable"] = float(r.iw_gain[-1][-1]) if len(r.iw_gain) else np.nan
    traj = [(q[0], q[-1]) for q in r.qshift]
    if decay is not None:
        out["err32_decay"] = float(decay.err_last[-1])
        out["errmean_decay"] = float(decay.err_mean[-1])
        out["q_decay"] = [float(v) for v in decay.qshift[-1]]
        out["q1_decay"] = float(decay.qshift[-1][-1])
        out["gain_decay"] = [float(v) for v in decay.iw_gain[-1]]
        out["gain_weak_decay"] = float(decay.iw_gain[-1][-1])
        traj = [(q[0], q[-1]) for q in decay.qshift]
    out["q_traj"] = traj
    # shifted-query error at matched in-distribution error (first crossing, log-linear interpolation)
    m = []
    for lv in LEVELS:
        v = np.nan
        for (a0, a1), (b0, b1) in zip(traj[:-1], traj[1:]):
            if a0 > lv >= b0 and a0 > 0 and b0 > 0:
                w = (np.log(a0) - np.log(lv)) / (np.log(a0) - np.log(b0))
                v = float(np.exp(np.log(a1) + w * (np.log(b1) - np.log(a1))))
                break
        m.append(v)
    out["q1_matched"] = m
    return out


def _f(x):
    return f"{x:,.0f}" if abs(x) >= 100 else f"{x:.3g}"


def agg(vals):
    v = np.array([x for x in vals if x is not None and np.isfinite(x)], float)
    n_missing = sum(1 for x in vals if x is None or not np.isfinite(x))
    if not len(v):
        return "never" if n_missing else "–"
    tail = f" (+{n_missing} never)" if n_missing else ""
    if len(v) == 1:
        return _f(v[0]) + tail
    return f"{_f(v.mean())} [{_f(v.min())}–{_f(v.max())}]" + tail


def main():
    root, outdir = sys.argv[1], sys.argv[2]
    os.makedirs(outdir, exist_ok=True)
    groups = defaultdict(list)
    for f in sorted(glob.glob(os.path.join(root, "*.json"))):
        if f.endswith("_decay.json"):
            continue
        st = load(f)
        if st.status != "done":
            continue
        df = f.replace(".json", "_decay.json")
        dc = load(df) if os.path.exists(df) and json.load(open(df))["status"] == "done" else None
        groups[(st.cond, st.opt)].append(run_metrics(st, dc))
    json.dump({f"{c}|{o}": v for (c, o), v in groups.items()}, open(os.path.join(outdir, "confirm.json"), "w"), default=float)
    cols = [("T_escape", "escape"), ("T_half", "half-way"), ("T_conv", "converged"), ("T_weak", "weakest axis"),
            ("axis_ratio", "axis8/axis1"), ("pcr_over_ridge", "PCR/ridge"), ("err32_stable", "err@32 stable"),
            ("err32_decay", "err@32 decayed"), ("q1_decay", "shifted-q decayed"), ("gain_weak_decay", "weak-axis gain")]
    print("| covariance | optimizer | n | " + " | ".join(c[1] for c in cols) + " |")
    print("|---|---|---|" + "---|" * len(cols))
    for c in CONDS:
        for o in ("adamw", "muon"):
            rs = groups.get((c, o), [])
            if not rs:
                continue
            print(f"| {c} | {o} | {len(rs)} | " + " | ".join(agg([r.get(k) for r in rs]) for k, _ in cols) + " |")
    print("\nShifted-query error (t = 1) at matched in-distribution error", LEVELS)
    for c in CONDS[1:]:
        for o in ("adamw", "muon"):
            rs = groups.get((c, o), [])
            if rs:
                print(f"  {c:9s} {o:6s} " + "  ".join(f"@{lv}: {agg([r['q1_matched'][i] for r in rs])}" for i, lv in enumerate(LEVELS)))
    a = groups.get(("1/i2-ali", "adamw"), [])
    b = groups.get(("1/i2-rot", "adamw"), [])
    if a and b:
        print("\nAdamW aligned vs rotated (1/i^2):")
        for k in ("T_conv", "T_weak", "err32_stable", "err32_decay", "q1_decay"):
            print(f"  {k:14s} aligned {agg([r.get(k) for r in a])}   rotated {agg([r.get(k) for r in b])}")


if __name__ == "__main__":
    main()
