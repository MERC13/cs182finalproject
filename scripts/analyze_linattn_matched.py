"""Linear attention (lambda ~ 1/i^2, d = 8): PCR-vs-ridge shape of the learned gain profile at MATCHED excess loss.

  python3 scripts/analyze_linattn_matched.py

Same runs and learning-rate selection as scripts/analyze_schatten_pilot.py (3 seeds per arm at the selected rate),
re-run with every step logged by scripts/linattn_dense.py (runs/linattn/dense_1i2.json; falls back to the original
10-step logs if that file is missing).
For each run, the gain profile g_j = clip(rho_j / rho_j^*, 0, 1.2) is interpolated (linearly between logged steps)
at 8 log-spaced excess-loss levels from 0.5 to 0.1 that every arm reaches, and compared with
  a PCR step     1[j <= m],            m = 0..d             -> R_PCR   = min_m sum_j (g_j - 1[j<=m])^2
  a ridge curve  lam_j / (lam_j + a),  200 log-spaced a in [1e-3, 1e2]  -> R_ridge = min_a sum_j (...)^2
Each residual is averaged over the levels (not over training time), and the ratio R_PCR / R_ridge is taken per
seed (> 1: closer to ridge). Also reported: the profile at excess loss 0.1. Writes figures/noise/linattn_matched.json.
"""
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
from analyze_linattn import metrics  # noqa: E402

DENSE = "runs/linattn/dense_1i2.json"  # scripts/linattn_dense.py: same runs, logged every step, 6,000 steps
BASE = ["runs/linattn/noise_sweep.json", "runs/linattn/noise_sweep_muonvar_exact.json"]
PILOT = "runs/linattn/schatten_pilot.json"
COND = "1/i2-ali"
ORDER = ["gd", "adam", "muon_pow1", "muon_pow0.75", "muon_pow0.5", "muon_pow0.25", "muon_pow0.1", "muon_exact", "muon"]
LEVELS = np.logspace(np.log10(0.5), np.log10(0.1), 8)
ALPHAS = np.logspace(-3, 2, 200)


def runs():
    if os.path.exists(os.path.join(ROOT, DENSE)):
        return [r for r in json.load(open(os.path.join(ROOT, DENSE))) if r["opt"] in ORDER]
    rs = []
    for f in BASE:
        rs += [r for r in json.load(open(os.path.join(ROOT, f))) if r.get("selected") and r["cond"] == COND
               and r["batch"] == 1024 and r.get("muon_momentum") in (None, 0.95) and r["opt"] in ORDER]
    rs += [r for r in json.load(open(os.path.join(ROOT, PILOT))) if r.get("selected") and r["cond"] == COND]
    return rs


def profile_at(m, lv):
    exc, rho = m["exc"], m["rho"]
    i = np.nonzero(exc <= lv)[0]
    if not len(i):
        return None
    i = int(i[0])
    if i == 0:
        return rho[0]
    t = (exc[i - 1] - lv) / (exc[i - 1] - exc[i])
    return rho[i - 1] + t * (rho[i] - rho[i - 1])


def residuals(g, lam):
    d = len(lam)
    r_pcr = min(np.sum((g - (np.arange(d) < k)) ** 2) for k in range(d + 1))
    shapes = lam[None, :] / (lam[None, :] + ALPHAS[:, None])
    r_ridge = np.min(np.sum((g[None, :] - shapes) ** 2, 1))
    return r_pcr, r_ridge


def main():
    out = {}
    print(f"levels: {np.round(LEVELS, 3)}")
    print(f"{'arm':14s} {'lr':>6s} n  ratio per seed (PCR/ridge, levels averaged)   median  [min-max]   profile at 0.1 (median)")
    for o in ORDER:
        rs = [r for r in runs() if r["opt"] == o]
        if not rs:
            continue
        ratios, profs, nlog = [], [], []
        for r in rs:
            m = metrics(r)
            nlog.append(int(((m["exc"] <= LEVELS[0]) & (m["exc"] >= LEVELS[-1])).sum()))
            lam = np.array(r["lam"])
            res = [residuals(np.clip(g, 0, 1.2), lam) for g in (profile_at(m, lv) for lv in LEVELS) if g is not None]
            if len(res) < len(LEVELS):
                ratios.append(float("nan"))
            else:
                ratios.append(float(np.mean([a for a, _ in res]) / np.mean([b for _, b in res])))
            p = profile_at(m, 0.1)
            if p is not None:
                profs.append(np.asarray(p))
        R = np.array(ratios)
        prof_med = np.median(profs, 0) if profs else np.full(8, np.nan)
        prof_lo = np.min(profs, 0) if profs else prof_med
        prof_hi = np.max(profs, 0) if profs else prof_med
        print(f"{o:14s} {rs[0]['lr']:6g} {len(rs)}  " + " ".join(f"{x:6.2f}" for x in R)
              + f"   {np.nanmedian(R):6.2f}  [{np.nanmin(R):.2f}-{np.nanmax(R):.2f}]   " + " ".join(f"{v:.2f}" for v in prof_med)
              + f"   | logged profiles in window {nlog}; mean gain dirs 3-8 at 0.1 {np.mean(prof_med[2:]):.3f}")
        out[o] = dict(lr=rs[0]["lr"], ratios=R.tolist(), median=float(np.nanmedian(R)), lo=float(np.nanmin(R)),
                      hi=float(np.nanmax(R)), profile=prof_med.tolist(), profile_lo=prof_lo.tolist(),
                      profile_hi=prof_hi.tolist(), n_profiles=len(profs))
    json.dump(dict(levels=LEVELS.tolist(), arms=out), open(os.path.join(ROOT, "figures/noise/linattn_matched.json"), "w"),
              indent=1)


if __name__ == "__main__":
    main()
