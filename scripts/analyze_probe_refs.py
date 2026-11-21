"""Probe profiles of the reference estimators, and a bounded version of the H1 probe score (Garg scale).

  python3 scripts/analyze_probe_refs.py

The probe score f_j = 1 - relative error on prompts whose task lies along direction j (inputs as in training) has no
lower bound. An estimator that ignores direction j still fits the probe's y with the directions it uses, so its
predictions there are noise and f_j < 0. This script
  1. computes f_j for the reference estimators (PCR_m, ridge_alpha, OLS) on the SAME probe prompts the models are
     evaluated on (eval seed 30000, 64 prompts per direction, averaged over predictions with 16-40 examples), and
  2. recomputes the H1 statistic with f_j clipped at 0 (mean of max(f_j, 0) over directions 3-20), per run at the
     first evaluation at or below each matched error, with paired one-sided t-tests between optimizers.
Writes figures/report/probe_refs.json.
"""
import json
import os
import sys

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
from icl.analysis import load  # noqa: E402
from icl.data import make_condition, make_eval_sets  # noqa: E402
from icl.estimators import RIDGE_GRID, ols, pcr, prefix_preds, ridge  # noqa: E402
from analyze_garg_seeds import ARMS, SEEDS, p_one_sided  # noqa: E402

D, N_POINTS, EVAL_SEED, K0 = 20, 41, 30000, 16
LEVELS = (0.1, 0.05, 0.04, 0.03)


def ref_profiles():
    cond = make_condition("1/i2-ali", D)
    ev = make_eval_sets(cond, N_POINTS, EVAL_SEED, n_probe=64)
    X, Y, pdir = ev.px.double(), ev.py.double(), ev.pdir
    U = torch.tensor(cond.U, dtype=torch.float64)
    lam = torch.tensor(cond.lam, dtype=torch.float64)
    ests = {"ols": ols}
    ests.update({f"ridge_{a:g}": ridge(a) for a in RIDGE_GRID})
    ests.update({f"pcr_{m}": pcr(U[:, :m], float(lam[m:].sum())) for m in range(1, D + 1)})
    out = {}
    for name, f in ests.items():
        pred = prefix_preds(X, Y, f)  # (P, n): prediction for y_k from k examples
        rel = torch.stack([((pred[pdir == j] - Y[pdir == j]) ** 2).sum(0) / (Y[pdir == j] ** 2).sum(0)
                           for j in range(D)])  # (d, n)
        out[name] = (1 - rel[:, K0:].mean(1)).tolist()
    return out


def run_profiles():
    prof = {}
    for name, pat in ARMS:
        for s in SEEDS:
            p = os.path.join(ROOT, "runs/garg_scale", pat.format(s))
            if not os.path.exists(p):
                continue
            r = load(p)
            for lv in LEVELS:
                i = np.nonzero(r.err_last <= lv)[0]
                if len(i):
                    prof[name, s, lv] = (1 - r.probe_err()[int(i[0])]).tolist()
    return prof


def main():
    refs = ref_profiles()
    np.set_printoptions(precision=2, suppress=True, linewidth=220)
    print("== reference estimators on the probe prompts: f_j for j = 1..20; mean over 3-20 raw and clipped at 0")
    for name in ["ols", "ridge_0.1", "ridge_1", "ridge_10", "ridge_100", "pcr_8", "pcr_10", "pcr_12", "pcr_14"]:
        f = np.array(refs[name])
        print(f"  {name:10s} raw {f[2:].mean():+.3f} clipped {np.clip(f[2:], 0, None).mean():.3f}  min {f.min():+.2f} ", f)
    prof = run_profiles()
    print("\n== H1 statistic per optimizer (mean over seeds), raw and clipped at 0")
    for lv in LEVELS:
        print(f"  error {lv}")
        for name, _ in ARMS:
            v = [np.array(prof[name, s, lv]) for s in SEEDS if (name, s, lv) in prof]
            if v:
                raw = [x[2:].mean() for x in v]
                clp = [np.clip(x[2:], 0, None).mean() for x in v]
                print(f"    {name:12s} n={len(v)} raw {np.mean(raw):+.3f} +- {np.std(raw, ddof=1):.3f}"
                      f"   clipped {np.mean(clp):.3f} +- {np.std(clp, ddof=1):.3f}")
        for a, b in (("Muon", "AdamW"), ("Muon", "SGD"), ("Muon", "Muon, p = 1")):
            ss = [s for s in SEEDS if (a, s, lv) in prof and (b, s, lv) in prof]
            if len(ss) >= 2:
                dd = [np.clip(prof[a, s, lv][2:], 0, None).mean() - np.clip(prof[b, s, lv][2:], 0, None).mean()
                      for s in ss]
                print(f"    clipped {a} - {b}: " + " ".join(f"{x:+.3f}" for x in dd)
                      + f"  mean {np.mean(dd):+.3f}  one-sided p = {p_one_sided(dd):.3f}")
    print("\n== mean profiles at error 0.04 (directions 1-20)")
    for name, _ in ARMS:
        v = np.array([prof[name, s, 0.04] for s in SEEDS if (name, s, 0.04) in prof])
        print(f"  {name:12s}", v.mean(0))
    os.makedirs(os.path.join(ROOT, "figures/report"), exist_ok=True)
    json.dump(dict(refs=refs, runs={f"{k[0]}|{k[1]}|{k[2]}": v for k, v in prof.items()}),
              open(os.path.join(ROOT, "figures/report/probe_refs.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
