"""Summary table for the GPT-2 main grid.  python scripts/analyze_grid.py 'runs/grid/*.json' figures/grid"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from icl.analysis import COND_ORDER, OPT_ORDER, best_fit, load_all  # noqa: E402

N_CTX = 32


def variant(r):
    v = r.opt
    if r.cfg.get("muon_params") == "hidden+in":
        v += "+in"
    if r.cfg.get("clip"):
        v += "/clip"
    return v


def path_shape(r):
    """Implied-weight gain profile vs PCR-step and ridge-shape families, averaged over the
    learning path (checkpoints whose excess error is between 5% and 90% of the plateau)."""
    if not len(r.iw_gain):
        return np.nan, np.nan
    lam = r.lam
    exc_at = np.interp(r.q_steps[-len(r.iw_gain):], r.steps, r.excess_mean())
    plateau = r.excess_mean()[0]
    idx = np.nonzero((exc_at > 0.05 * plateau) & (exc_at < 0.9 * plateau))[0]
    if len(idx) == 0:
        return np.nan, np.nan
    alphas = np.logspace(-2, 3, 300)
    ridge = N_CTX * lam[None] / (N_CTX * lam[None] + alphas[:, None])
    pr = []
    for j in idx:
        g = np.clip(r.iw_gain[j], -0.2, 1.2)
        pcr = min(np.sum((g - (np.arange(len(lam)) < m)) ** 2) for m in range(len(lam) + 1))
        rid = np.min(np.sum((g[None] - ridge) ** 2, 1))
        pr.append((pcr, rid))
    pr = np.array(pr)
    return pr[:, 0].mean(), pr[:, 1].mean()


def dist_shape(r, which="dist_8_32"):
    """Median over the learning path of (best PCR-m distance) / (best ridge-lambda distance),
    from the model-vs-estimator prediction distances logged every eval (k = 8..32 by default).
    < 1: predictions are closer to some PCR(m) than to any ridge; > 1: ridge-like."""
    D = getattr(r, "dist8" if which == "dist_8_32" else "dist")
    if D is None:
        return np.nan, []
    exc = r.excess_mean()
    plateau = exc[0]
    idx = np.nonzero((exc > 0.05 * plateau) & (exc < 0.9 * plateau))[0]
    if len(idx) == 0:
        return np.nan, []
    P = np.stack([D[k] for k in D if k.startswith("pcr_")])
    R = np.stack([D[k] for k in D if k.startswith("ridge_")])
    ratio = P.min(0)[idx] / R.min(0)[idx]
    mstar = [int(np.argmin(P[:, j])) + 1 for j in idx]
    return float(np.median(ratio)), mstar


def summarize(r):
    exc = r.excess_mean()
    ols_mean = r.ref_err["ols"][1:].mean()
    plateau = 1.0 - ols_mean
    out = dict(run=os.path.basename(r.path), opt=variant(r), cond=r.cond, seed=r.cfg["seed"], lr=r.lr,
               status=r.status, steps=int(r.steps[-1]),
               T_escape=r.first_below(r.err_mean, 0.9),
               T_half=r.first_below(exc, 0.5 * plateau),
               T_conv=r.first_below(exc, 0.05),
               err_mean_final=float(r.err_mean[-1]), excess_final=float(exc[-1]), err32_final=float(r.err_last[-1]))
    if r.probe is not None:
        T_ax = r.axis_learn_steps(0.5)
        out["T_axis"] = T_ax.tolist()
        out["probe_final"] = r.probe_err()[-1].tolist()
    if len(r.iw_gain):
        out["iw_gain_final"] = r.iw_gain[-1].tolist()
        out["iw_r2_final"] = float(r.iw_r2[-1])
    if len(r.qshift):
        out["qshift_final"] = r.qshift[-1].tolist()
        out["q_steps"] = r.q_steps.tolist()
        out["qshift_all"] = r.qshift.tolist()
    if r.dist is not None:
        mr, dr = best_fit(r, "ridge_")
        mp, dp = best_fit(r, "pcr_")
        mg, dg = best_fit(r, "gdpre_")
        out["best_fit_final"] = dict(ridge=(mr[-1], float(dr[-1])), pcr=(mp[-1], float(dp[-1])),
                                     gdpre=(mg[-1], float(dg[-1])), ols=float(r.dist["ols"][-1]))
    out["pcr_res"], out["ridge_res"] = path_shape(r)
    out["pcr_over_ridge"], out["mstar_path"] = dist_shape(r)
    out["n_path"] = len(out["mstar_path"])
    return out


def fmt(v, w=7, p=0):
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return f"{'-':>{w}s}"
    return f"{v:{w}.{p}f}"


if __name__ == "__main__":
    pattern, outdir = sys.argv[1], sys.argv[2]
    os.makedirs(outdir, exist_ok=True)
    runs = load_all(pattern)
    rows = [summarize(r) for r in runs]
    order = {o: i for i, o in enumerate(["sgd", "sgd/clip", "adamw", "muon", "muon+in"])}
    rows.sort(key=lambda s: (COND_ORDER.index(s["cond"]), order.get(s["opt"], 9), s["seed"]))
    print(f"{'cond':9s} {'opt':9s} {'s':>1s} {'lr':>7s} {'steps':>6s} {'T_esc':>6s} {'T_half':>6s} {'T_conv':>6s} "
          f"{'exc_fin':>7s} {'err32':>6s} {'q(t=1)':>7s} {'PCRres':>6s} {'RIDres':>6s} {'P/R':>5s} {'n':>3s}  T_axis(probe<0.5) / iw gain final")
    for s in rows:
        q1 = s["qshift_final"][-1] if "qshift_final" in s else np.nan
        tax = " ".join(fmt(t, 5) for t in s.get("T_axis", []))
        gains = " ".join(f"{g:.2f}" for g in s.get("iw_gain_final", []))
        print(f"{s['cond']:9s} {s['opt']:9s} {s['seed']:1d} {s['lr']:7.2g} {s['steps']:6d} {fmt(s['T_escape'], 6)} "
              f"{fmt(s['T_half'], 6)} {fmt(s['T_conv'], 6)} {fmt(s['excess_final'], 7, 3)} {fmt(s['err32_final'], 6, 3)} "
              f"{fmt(q1, 7, 3)} {fmt(s['pcr_res'], 6, 3)} {fmt(s['ridge_res'], 6, 3)} {fmt(s['pcr_over_ridge'], 5, 2)} {s['n_path']:3d}  [{tax}] / [{gains}]  {s['status']}")
    json.dump(rows, open(os.path.join(outdir, "summary.json"), "w"), indent=1, default=float)
