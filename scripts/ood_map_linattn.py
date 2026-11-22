"""When does Muon help out of distribution?  One-layer linear attention, exact losses, no new training.

  python scripts/ood_map_linattn.py

The predictor is yhat = beta^T Gamma x_q with beta = (1/N) sum_n y_n x_n. For contexts x_n ~ N(0, C), query
x_q ~ N(0, Q), tasks w ~ N(0, W), label noise sigma^2 and N context examples, its expected squared error is
  L = (N+1)/N tr(C A C W) + tr(A C) tr(C W)/N + sigma^2 tr(A C)/N - 2 tr(C Gamma Q W) + tr(Q W),  A = Gamma Q Gamma^T
(Gaussian moments; with C = Q = Lambda, W = I, sigma = 0 it is the testbed's own pop()). Gamma is the full learned
matrix logged by scripts/ood_rerun_linattn.py (checked against the logged losses below).

Every run is read at the first step where its in-distribution excess loss reaches LEVEL, and each shift family is
then applied to that predictor. Ratios < 1: the arm beats Adam under that shift.
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from analyze_linattn import metrics  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FILES = {"1/i2-ali": ["runs/linattn/ood_gamma_dense.json"],  # Gamma logged every step near the levels (linattn_dense_gamma.py)
         "1/i-ali": ["runs/linattn/ood_gamma.json"]}  # Gamma logged every 10 steps (ood_rerun_linattn.py)
ARMS = ["gd", "adam", "muon", "muon_pow1", "muon_pow0.5", "muon_pow0.25", "muon_pow0.1", "muon_exact"]
NAME = {"gd": "SGD", "adam": "Adam", "muon": "Muon (NS)", "muon_exact": "p=0 exact"}
N, D = 32, 8
LEVELS = (0.1, 0.05)


def loss(G, c, q, s, sigma2=0.0, n=N):
    """Exact error of yhat = beta^T G x_q (G in the principal basis) for C = diag(c), Q = diag(q), W = diag(s)."""
    C, Q, W = np.diag(c), np.diag(q), np.diag(s)
    A = G @ Q @ G.T
    t1 = (n + 1) / n * np.trace(C @ A @ C @ W)
    t2 = np.trace(A @ C) * np.trace(C @ W) / n
    t3 = sigma2 * np.trace(A @ C) / n
    t4 = -2 * np.trace(C @ G @ Q @ W)
    t5 = np.trace(Q @ W)
    return (t1 + t2 + t3 + t4 + t5) / D


def norm(v, total):
    return v * total / v.sum()


def shifts(lam):
    T = lam.sum()
    one = np.ones_like(lam)
    pw = lambda a: norm(lam ** a, T)
    S = {}
    S["in-distribution"] = dict(c=lam, q=lam, s=one)
    # 1. only the query moves (the proposal's test is a = -1)
    for a in (3, 2, 0, -1):
        S[f"query ~ lam^{a}"] = dict(c=lam, q=pw(a), s=one)
    # 2. the whole prompt moves: context and query from the same new covariance
    for a in (3, 2, 0, -1):
        S[f"prompt ~ lam^{a}"] = dict(c=pw(a), q=pw(a), s=one)
    # 3. only the context moves
    for a in (0, -1):
        S[f"context ~ lam^{a}"] = dict(c=pw(a), q=lam, s=one)
    # 4. tasks move: w ~ N(0, diag(s)), same total task variance
    for a in (2, 1, -1, -2):
        S[f"tasks ~ lam^{a}"] = dict(c=lam, q=lam, s=norm(lam ** a, len(lam)))
    # 5. label noise at test (signal variance tr(Lambda) = T)
    for snr in (4, 1):
        S[f"label noise, SNR {snr}"] = dict(c=lam, q=lam, s=one, sigma2=T / snr)
    # 6. context length at test
    for n in (8, 16, 128):
        S[f"context length {n}"] = dict(c=lam, q=lam, s=one, n=n)
    return S


def losses_at_level(m, G, lv, S):
    """Each shift's loss at the first crossing of in-distribution excess loss lv, interpolated between the two
    logged steps around it with the same weight as the in-distribution loss (so every arm is matched exactly)."""
    exc = m["exc"]
    i = np.nonzero(exc <= lv)[0]
    if not len(i):
        return None
    i = int(i[0])
    if i == 0:
        return {k: loss(G[0], **v) for k, v in S.items()}
    t = (exc[i - 1] - lv) / (exc[i - 1] - exc[i])
    return {k: (1 - t) * loss(G[i - 1], **v) + t * loss(G[i], **v) for k, v in S.items()}


def at_level(m, G, lv):
    """Gamma at the first crossing of in-distribution excess loss lv (linear in Gamma between logged steps)."""
    exc = m["exc"]
    i = np.nonzero(exc <= lv)[0]
    if not len(i):
        return None
    i = int(i[0])
    if i == 0:
        return G[0]
    t = (exc[i - 1] - lv) / (exc[i - 1] - exc[i])
    return G[i - 1] + t * (G[i] - G[i - 1])


def main():
    out = {}
    for cond, files in FILES.items():
        runs = []
        for f in files:
            runs += [r for r in json.load(open(os.path.join(ROOT, f))) if r.get("selected") and r["cond"] == cond
                     and r["batch"] == 1024 and r.get("muon_momentum") in (None, 0.95) and r["opt"] in ARMS]
        lam = np.array(runs[0]["lam"])
        rs = np.array(runs[0]["rho_star"])
        # check the diagonal approximation on every logged step of every run
        err_L, err_Q = 0.0, 0.0
        T = lam.sum()
        qinv = norm(lam ** -1.0, T)
        for r in runs:
            G = np.array(r["gamma"])
            for k in range(0, len(r["steps"]), 20):
                err_L = max(err_L, abs(loss(G[k], lam, lam, np.ones_like(lam)) - r["loss"][k]))
                err_Q = max(err_Q, abs(loss(G[k], lam, qinv, np.ones_like(lam)) - r["qloss"][k]))
        print(f"\n== {cond}: formula check, max |error| vs logged loss {err_L:.2e}, vs logged shifted-query loss {err_Q:.2e}")
        S = shifts(lam)
        for lv in LEVELS:
            res = {}
            for o in ARMS:
                vals = []
                for r in runs:
                    if r["opt"] != o:
                        continue
                    v = losses_at_level(metrics(r), np.array(r["gamma"]), lv, S)
                    if v is not None:
                        vals.append(v)
                if vals:
                    res[o] = {k: float(np.median([v[k] for v in vals])) for k in S}
                    res[o]["n"] = len(vals)
            out[f"{cond}|{lv}"] = res
            print(f"\n-- matched in-distribution excess loss {lv}: loss under each shift (median over seeds); "
                  f"in brackets the ratio to Adam (< 1 = better than Adam)")
            cols = [o for o in ARMS if o in res]
            print(f"{'shift':26s} " + " ".join(f"{NAME.get(o, o.replace('muon_pow', 'p=')):>15s}" for o in cols))
            for k in S:
                base = res["adam"][k] if "adam" in res else np.nan
                print(f"{k:26s} " + " ".join(f"{res[o][k]:7.3f} ({res[o][k] / base:4.2f})" for o in cols))
            print(f"{'seeds':26s} " + " ".join(f"{res[o]['n']:>15d}" for o in cols))
    json.dump(out, open(os.path.join(ROOT, "figures/noise/ood_map_linattn.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
