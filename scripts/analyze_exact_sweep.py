"""Robustness of the testbed conclusions: exact population gradients, dense LR grids, two init scales.

For each (parametrization, init, covariance, optimizer): best convergence time over the LR grid
(steps until excess loss < 5%), the LR achieving it, the weakest/strongest-axis time ratio at that
LR, and how many grid LRs converge at all.

  python scripts/analyze_exact_sweep.py runs/linattn/exact_sweep.json
"""
import json
import sys
from collections import defaultdict

import numpy as np

R = json.load(open(sys.argv[1]))
G = defaultdict(list)
for r in R:
    G[(r["mode"], r["init"], r["cond"], r["opt"])].append(r)
OPTS = ["gd", "signgd", "adam", "muon", "muon_exact"]
for mode in ("sep", "merged"):
    for init in (1e-3, 1e-2):
        print(f"\n== {mode} K,Q  init {init:g}   best steps-to-converge over LR grid (ratio = weakest/strongest axis time; n = #LRs converged)")
        print(f"   {'cond':9s} " + " ".join(f"{o:>22s}" for o in OPTS))
        for cond in ("iso", "1/i2-ali", "1/i2-rot"):
            cells = []
            for o in OPTS:
                rs = [r for r in G.get((mode, init, cond, o), []) if r["status"] == "done"]
                conv = [r for r in rs if np.isfinite(r["T_conv"])]
                if not conv:
                    cells.append(f"{'> budget':>22s}")
                    continue
                b = min(conv, key=lambda r: r["T_conv"])
                ta = np.array(b["T_axis"], float)
                ratio = ta[-1] / ta[0] if np.isfinite(ta).all() and ta[0] > 0 else float("nan")
                cells.append(f"{b['T_conv']:6.0f} (x{ratio:4.1f}, n={len(conv)})".rjust(22))
            print(f"   {cond:9s} " + " ".join(cells))
