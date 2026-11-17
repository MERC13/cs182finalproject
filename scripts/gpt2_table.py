"""Markdown table of the GPT-2 grid from figures/grid/summary.json (seed 0, plus seed-1 values in brackets)."""
import json
import sys

import numpy as np

S = json.load(open(sys.argv[1]))
COND = ["iso", "1/i-ali", "1/i2-ali", "1/i2-rot"]
CL = {"iso": "isotropic", "1/i-ali": "1/i aligned", "1/i2-ali": "1/i² aligned", "1/i2-rot": "1/i² rotated"}
OPT = [("sgd/clip", "SGD"), ("adamw", "AdamW"), ("muon", "Muon"), ("muon+in", "Muon + read-in")]


def f(v, p=0):
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return "–"
    return f"{v:,.{p}f}"


def cell(rows, key, p=0, fn=None):
    vals = [(fn(r) if fn else r.get(key)) for r in sorted(rows, key=lambda r: r["seed"])]
    if not vals:
        return ""
    s = f(vals[0], p)
    if len(vals) > 1:
        s += " (" + ", ".join(f(v, p) for v in vals[1:]) + ")"
    return s


print("| covariance | optimizer | leaves plateau | half-way | converged | weakest axis learned | axis-8/axis-1 time | in-dist. error @32 | shifted-query error (t=1) | PCR/ridge fit ratio |")
print("|---|---|---|---|---|---|---|---|---|---|")
for c in COND:
    for o, lab in OPT:
        rows = [r for r in S if r["cond"] == c and r["opt"] == o and r["status"] == "done"]
        if not rows:
            continue
        never = lambda r: (r["T_axis"][-1] if "T_axis" in r else None)
        ratio = lambda r: (r["T_axis"][-1] / r["T_axis"][0] if "T_axis" in r and np.isfinite(r["T_axis"][-1]) and np.isfinite(r["T_axis"][0]) else None)
        q1 = lambda r: r["qshift_final"][-1] if "qshift_final" in r else None
        print(f"| {CL[c]} | {lab} | {cell(rows, 'T_escape')} | {cell(rows, 'T_half')} | {cell(rows, 'T_conv')} | "
              f"{cell(rows, None, fn=never)} | {cell(rows, None, 1, fn=ratio)} | {cell(rows, 'err32_final', 3)} | "
              f"{cell(rows, None, 3, fn=q1)} | {cell(rows, 'pcr_over_ridge', 2)} |")
