"""Analyse the spectral-flattening pilot (scripts/schatten_pilot_linattn.py) next to the batch-1,024 baselines.

  python scripts/analyze_schatten_pilot.py

Per arm (SGD, Adam, Muon with singular values raised to p = 1 ... 0, standard Newton-Schulz Muon), median
over seeds at the selected learning rate:
  shape     fraction of each direction's optimal gain learned when the in-distribution excess loss first
            reaches 0.1; "weak" = its mean over directions 3-8 (about 18% of the input variance under 1/i^2)
  ridge-fit PCR residual / ridge residual of the gain profile along the learning path (> 1: closer to ridge)
  q@0.05    shifted-query excess loss when the in-distribution excess loss first reaches 0.05, minus the best
            converged shifted-query loss of the earlier redesign analysis (as in the proposal's table)
  beta      learning-order exponent; T_conv steps to excess loss 0.05
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from analyze_linattn import metrics  # noqa: E402
from analyze_noise import beta  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASE = {"1/i2-ali": ["runs/linattn/noise_sweep.json", "runs/linattn/noise_sweep_muonvar_exact.json"],
        "1/i-ali": ["runs/linattn/noise_sweep_1i.json"]}
PILOT = "runs/linattn/schatten_pilot.json"
ORDER = ["gd", "adam", "muon_pow1", "muon_pow0.75", "muon_pow0.5", "muon_pow0.25", "muon_pow0.1", "muon_exact", "muon"]
NAME = {"gd": "SGD", "adam": "Adam", "muon_exact": "p = 0 (exact Muon)", "muon": "Muon (Newton-Schulz)"}


def at(m, key, lv):
    exc, v = m["exc"], m[key]
    i = np.nonzero(exc <= lv)[0]
    if not len(i):
        return None
    i = int(i[0])
    if i == 0:
        return v[0]
    t = (exc[i - 1] - lv) / (exc[i - 1] - exc[i])
    return v[i - 1] + t * (v[i] - v[i - 1])


def main():
    out = {}
    for cond, files in BASE.items():
        base = []
        for f in files:
            base += [r for r in json.load(open(os.path.join(ROOT, f))) if r.get("selected") and r["cond"] == cond
                     and r["batch"] == 1024 and r.get("muon_momentum") in (None, 0.95) and r["opt"] in ORDER]
        floor = min(float(metrics(r)["q"][-1]) for r in base)  # same floor as the redesign analysis' batch-1,024 rows
        pilot = [r for r in json.load(open(os.path.join(ROOT, PILOT))) if r.get("selected") and r["cond"] == cond]
        allr = base + pilot
        print(f"\n== {cond}: shifted-query floor {floor:.4f}; lowest final shifted-query loss in the pilot "
              f"{min(float(metrics(r)['q'][-1]) for r in pilot):.4f}")
        print(f"{'arm':22s} {'lr':>6s} {'n':>2s} | {'gain profile at excess 0.1 (directions 1-8)':47s} | {'weak':>5s} "
              f"{'ridge-fit':>9s} | {'q@0.05':>17s} | {'beta':>5s} {'T_conv':>6s}")
        for o in ORDER:
            rs = [r for r in allr if r["opt"] == o]
            if not rs:
                continue
            M = [metrics(r) for r in rs]
            P = [p for p in (at(m, "rho", 0.1) for m in M) if p is not None]
            prof = np.median(P, 0) if P else np.full(8, np.nan)
            Q = np.array([np.nan if (q := at(m, "q", 0.05)) is None else q - floor for m in M])
            fit = np.median([m["pcr_res"] / m["ridge_res"] for m in M])
            B = [beta(m["T_axis"], r["lam"]) for r, m in zip(rs, M)]
            T = [m["T_conv"] for m in M]
            name = NAME.get(o, f"p = {o[len('muon_pow'):]}")
            qs = "never" if np.all(np.isnan(Q)) else f"{np.nanmedian(Q):.3f} [{np.nanmin(Q):.3f}-{np.nanmax(Q):.3f}]"
            print(f"{name:22s} {rs[0]['lr']:6g} {len(rs):2d} | " + " ".join(f"{v:5.2f}" for v in prof)
                  + f" | {np.nanmean(prof[2:]):5.2f} {fit:9.2f} | {qs:>17s} | {np.nanmedian(B):5.2f} {np.nanmedian(T):6.0f}")
            out[f"{cond}|{o}"] = dict(lr=rs[0]["lr"], n=len(rs), profile=prof.tolist(), weak=float(np.nanmean(prof[2:])),
                                      ridge_fit=float(fit), q005=Q.tolist(), beta=B, T_conv=T,
                                      final_loss=[float(m["Lfinal"]) for m in M])
    os.makedirs(os.path.join(ROOT, "figures/noise"), exist_ok=True)
    json.dump(out, open(os.path.join(ROOT, "figures/noise/schatten_pilot.json"), "w"), indent=1, default=float)


if __name__ == "__main__":
    main()
