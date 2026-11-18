"""Do the proposal's hypotheses hold at Garg et al.'s model scale? (jobs/garg_scale.txt)

  python scripts/analyze_garg_scale.py [runs/garg_scale]

12 layers, 8 heads, width 256; d = 20, 40 examples; lambda ∝ 1/i², concatenated tokens. For every error level that all
runs reach (in-distribution error with 40 examples, relative to predicting 0):
  H1  fraction of each direction learned (1 - probe error) at the first evaluation at or below that level;
      "weak" = mean over all but the two strongest directions (about 22% of the input variance)
  H2  shifted-query error at the same matched level (first crossing of the in-distribution query error on the
      shifted-query prompts, log-linear interpolation), queries shifted toward the rare (t > 0) or common (t < 0)
      directions; ratio to AdamW
  H3  spectral knob: Muon with the update's singular values unflattened (p = 1) vs standard and exact (p = 0) Muon
"""
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from icl.analysis import load  # noqa: E402

DIR = sys.argv[1] if len(sys.argv) > 1 else "runs/garg_scale"
ARMS = [("SGD", "sgd_s10.json"), ("AdamW", "adamw_s10.json"), ("Muon", "muon_s10.json"), ("Muon, exact", "muon_exact_s10.json"),
        ("Muon, p = 1", "muon_pow1_lr0.0005_s10.json")]
LEVELS = (0.3, 0.2, 0.1, 0.05, 0.04, 0.03)


def q_matched(d, lv):
    ts = [float(t) for t in d["config"]["qshift_ts"].split(",")]
    Q = [np.array(e["qshift_err"]) for e in d["evals"] if "qshift_err" in e]
    i0 = ts.index(0.0)
    for a, b in zip(Q[:-1], Q[1:]):
        if a[i0] > lv >= b[i0] and a[i0] > 0 and b[i0] > 0:
            w = (np.log(a[i0]) - np.log(lv)) / (np.log(a[i0]) - np.log(b[i0]))
            return ts, np.exp(np.log(a) + w * (np.log(b) - np.log(a)))
    return ts, None


def main():
    runs = {}
    for name, f in ARMS:
        p = os.path.join(ROOT, DIR, f)
        if os.path.exists(p):
            runs[name] = (load(p), json.load(open(p)))
    print(f"{'run':12s} {'status':9s} {'last step':>9s} {'error@40':>8s} | first step at error " + " ".join(f"{lv:>6g}" for lv in LEVELS))
    for name, (r, d) in runs.items():
        first = []
        for lv in LEVELS:
            i = np.nonzero(r.err_last <= lv)[0]
            first.append(f"{int(r.steps[i[0]]):6d}" if len(i) else f"{'-':>6s}")
        print(f"{name:12s} {d.get('status', ''):9s} {int(r.steps[-1]):9d} {float(r.err_last[-1]):8.3f} | {' '.join(first)}")
    for lv in LEVELS:
        reached = {n: np.nonzero(r.err_last <= lv)[0] for n, (r, d) in runs.items()}
        names = [n for n in runs if len(reached[n])]
        if len(names) < 2:
            continue
        print(f"\n== matched in-distribution error {lv} (reached by: {', '.join(names)})")
        dd = len(runs[names[0]][0].probe_err()[0])
        show = sorted({j for j in (1, 2, 3, dd // 4, dd // 2, 3 * dd // 4, dd) if 1 <= j <= dd})
        print("H1, fraction learned:   " + " ".join(f"{'dir ' + str(j):>7s}" for j in show) + f" {'weak (3-' + str(dd) + ')':>12s}")
        for n in names:
            r = runs[n][0]
            prof = 1 - r.probe_err()[int(reached[n][0])]
            print(f"  {n:20s} " + " ".join(f"{prof[j - 1]:7.2f}" for j in show) + f" {np.mean(prof[2:]):12.3f}")
        base = q_matched(runs["AdamW"][1], lv)[1] if "AdamW" in names else None
        print("H2, shifted-query error (ratio to AdamW); t < 0 toward common directions, t > 0 toward rare ones")
        for n in names:
            ts, q = q_matched(runs[n][1], lv)
            if q is None:
                print(f"  {n:20s} shifted-query prompts did not cross this level")
                continue
            order = sorted(range(len(ts)), key=lambda k: ts[k])
            cells = [f"t={ts[k]:+g}: {q[k]:.3f}" + (f" ({q[k] / base[k]:.2f})" if base is not None else "") for k in order]
            print(f"  {n:20s} " + "  ".join(cells))


if __name__ == "__main__":
    main()
