"""The redesigned experiment in the one-layer linear-attention testbed (separate rank-1 K/Q, aligned covariance).

  python scripts/analyze_linattn_redesign.py

Per optimizer x batch size x spectrum (λ ∝ 1/i and 1/i²), selected LR, 3 seeds:
  beta    spectral learning exponent (time to learn axis j ∝ λ_j^-beta), learned axes only
  R3      query-shift loss (exact; queries on the inverted spectrum, same trace) at the first step where the
          in-distribution excess loss reaches 0.1 and 0.05, minus the best converged query-shift loss among
          all runs of that spectrum: 'excess query-shift loss at matched in-distribution error'.
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from analyze_linattn import metrics  # noqa: E402
from analyze_noise import beta  # noqa: E402

FILES = {"1/i2-ali": ["runs/linattn/noise_sweep.json", "runs/linattn/noise_sweep_muonvar_exact.json"],
         "1/i-ali": ["runs/linattn/noise_sweep_1i.json"]}
OPTS = ["gd", "adam", "muon", "muon_exact"]
LEVELS = (0.1, 0.05)


def q_at(m, lv):
    exc, q = m["exc"], m["q"]
    i = np.nonzero(exc <= lv)[0]
    if not len(i):
        return np.nan
    i = int(i[0])
    if i == 0:
        return float(q[0])
    t = (exc[i - 1] - lv) / (exc[i - 1] - exc[i])
    return float(q[i - 1] + t * (q[i] - q[i - 1]))


def main():
    out = {}
    for cond, files in FILES.items():
        R = []
        for f in files:
            if os.path.exists(f):
                R += [r for r in json.load(open(f)) if r.get("selected") and r["cond"] == cond
                      and r["opt"] in OPTS and (r.get("muon_momentum") in (None, 0.95))]
        if not R:
            continue
        M = [(r, metrics(r)) for r in R]
        floor = min(float(m["q"][-1]) for _, m in M)
        print(f"\n== {cond}: best converged query-shift loss {floor:.3f} (untrained 1.00)")
        print(f"{'optimizer':11s} {'batch':>5s} | {'beta (3 seeds)':>16s} | " +
              " | ".join(f"excess q-shift at ID excess {lv}" for lv in LEVELS))
        for o in OPTS:
            for b in (16, 64, 256, 1024):
                v = [(r, m) for r, m in M if r["opt"] == o and r["batch"] == b]
                if not v:
                    continue
                bs = [beta(m["T_axis"], r["lam"]) for r, m in v]
                qs = [[q_at(m, lv) - floor for lv in LEVELS] for _, m in v]
                Q = np.array(qs)
                out[f"{cond}|{o}|{b}"] = dict(beta=bs, q=qs, lr=v[0][0]["lr"])
                cell = lambda j: ("never" if np.all(np.isnan(Q[:, j])) else
                                  f"{np.nanmedian(Q[:, j]):.3f} [{np.nanmin(Q[:, j]):.3f}-{np.nanmax(Q[:, j]):.3f}]"
                                  + (f" ({int(np.sum(~np.isnan(Q[:, j])))}/{len(Q)})" if np.isnan(Q[:, j]).any() else ""))
                print(f"{o:11s} {b:5d} | {np.nanmedian(bs):5.2f} [{np.nanmin(bs):.2f}-{np.nanmax(bs):.2f}] | "
                      + " | ".join(f"{cell(j):>30s}" for j in range(len(LEVELS))))
    os.makedirs("figures/noise", exist_ok=True)
    json.dump(out, open("figures/noise/linattn_redesign.json", "w"), indent=1, default=float)


if __name__ == "__main__":
    main()
