"""Reproductions of the theory papers' claims, computed exactly (no sampling noise).

  zhang   Zhang et al. 2025: gradient flow on the exact population loss of 8-head linear attention
          (separate rank-1 K,Q and merged KQ) integrated with an adaptive ODE solver.  Checks the
          predicted plateau losses L_m = trL - sum_{i<=m} lam_i / (1 + (1 + trL/lam_i)/N), the
          eigenvalue order of learning, and the merged model's single drop to the optimum
          (von Oswald et al./Zhang et al.: one step of preconditioned GD, Gamma* = U diag(c) U^T).
  ma      Ma et al. 2026: f(Q) = 1/2 tr((SQ - I) S (SQ - I)^T), Q0 = 0.  Iterations to reach
          f/f(0) <= eps for GD (step 1/L), SignGD and simplified Muon (exact polar factor, no
          momentum) with geometric schedules eta_t = C/sigma_min(S) rho^t (grid-searched), versus
          kappa(S), for diagonal (aligned) and rotated S.
  deeplin Saxe et al./Dragutinovic et al.: two-layer linear network W2 W1 -> A = U diag(s) V^T from
          small init; time for each singular mode to be learned under GD, Adam and spectral GD.

  python -m icl.repro_theory --out figures/verify/repro_theory.json
"""
from __future__ import annotations

import argparse
import json
import math
import time

import numpy as np
import torch
from scipy.integrate import solve_ivp

from .data import haar_rotation, spectrum

torch.set_default_dtype(torch.float64)


# ----------------------------------------------------------------------------- Zhang et al.
def pop_loss(G, S, N):
    """Exact population loss / d of yhat = beta^T G x_q, beta = (1/N) sum y_n x_n (w ~ N(0, I))."""
    A = G @ S @ G.T
    ESAS = (N + 1) / N * S @ A @ S + torch.trace(A @ S) / N * S
    return (torch.trace(ESAS) - 2 * torch.trace(S @ G.T @ S) + torch.trace(S)) / S.shape[0]


def zhang_flow(lam, U, mode, N=32, H=8, init=5e-3, t_max=1e4, seed=0, n_eval=400):
    d = len(lam)
    S = torch.tensor(U @ np.diag(lam) @ U.T)
    g = torch.Generator().manual_seed(seed)
    if mode == "sep":
        shapes = [(H, d), (H, d), (H,)]
    else:
        shapes = [(H, d, d), (H,)]
    p0 = torch.cat([init * torch.randn(*s, generator=g).reshape(-1) for s in shapes])
    sizes = [int(np.prod(s)) for s in shapes]

    def gamma(p):
        parts = list(torch.split(p, sizes))
        if mode == "sep":
            K, Q, v = parts[0].view(H, d), parts[1].view(H, d), parts[2]
            return torch.einsum("h,hi,hj->ij", v, K, Q)
        Uh, v = parts[0].view(H, d, d), parts[1]
        return torch.einsum("h,hij->ij", v, Uh)

    def rhs(t, y):
        p = torch.tensor(y, requires_grad=True)
        L = pop_loss(gamma(p), S, N)
        (gp,) = torch.autograd.grad(L, p)
        return (-gp).numpy()

    ts = np.unique(np.clip(np.concatenate([[0.0], np.logspace(-2, math.log10(t_max), n_eval)]), 0.0, t_max))
    t0 = time.time()
    sol = solve_ivp(rhs, (0, t_max), p0.numpy(), t_eval=ts, method="LSODA", rtol=1e-7, atol=1e-10)
    Ut = torch.tensor(U)
    lam_t = torch.tensor(lam)
    T = float(lam_t.sum())
    c = N / ((N + 1) * lam_t + T)
    out = dict(t=sol.t.tolist(), loss=[], gain=[], wall=round(time.time() - t0, 1))
    for y in sol.y.T:
        G = gamma(torch.tensor(y))
        out["loss"].append(float(pop_loss(G, S, N)))
        out["gain"].append((lam_t * torch.diagonal(Ut.T @ G @ Ut) / (lam_t * c)).tolist())  # fraction of optimal gain
    Gstar = Ut @ torch.diag(c) @ Ut.T
    Gfin = gamma(torch.tensor(sol.y[:, -1]))
    out["final_rel_err_to_Gstar"] = float((Gfin - Gstar).norm() / Gstar.norm())
    Lm = [T]
    for i in range(len(lam)):
        Lm.append(Lm[-1] - lam[i] / (1 + (1 + T / lam[i]) / N))
    out["L_m_pred"] = [v / d for v in Lm]  # normalised by d like the loss above
    out["L_star"] = float(pop_loss(Gstar, S, N))
    return out


def plateau_check(res, thr=0.5):
    """Measured loss right before each axis i turns on (gain crosses thr) vs predicted L_{i-1}."""
    t, gain, loss = np.array(res["t"]), np.array(res["gain"]), np.array(res["loss"])
    rows = []
    for i in range(gain.shape[1]):
        on = np.nonzero(gain[:, i] > thr)[0]
        if not len(on):
            rows.append(dict(axis=i + 1, t_on=None, measured=None, predicted=res["L_m_pred"][i]))
            continue
        j = on[0]
        # plateau value: loss at the log-midpoint between the previous axis turning on and this one
        prev = 0 if i == 0 else rows[-1]["t_on"] or 0
        tm = math.sqrt(max(prev, t[1]) * t[j])
        k = int(np.argmin(np.abs(t - tm)))
        rows.append(dict(axis=i + 1, t_on=float(t[j]), measured=float(loss[k]), predicted=res["L_m_pred"][i]))
    return rows


# ----------------------------------------------------------------------------- Ma et al.
def ma_problem(d, kappa, rotated, seed=0):
    lam = kappa ** (-np.arange(d) / max(d - 1, 1))  # 1 ... 1/kappa
    U = haar_rotation(d, seed) if rotated else np.eye(d)
    return torch.tensor(U @ np.diag(lam) @ U.T), float(lam.min())


def ma_run(S, method, eta0=None, rho=None, eps=1e-6, max_it=5000):
    d = S.shape[0]
    Q = torch.zeros(d, d)
    I = torch.eye(d)
    f0 = 0.5 * torch.trace(S).item()
    Lmax = torch.linalg.eigvalsh(S).max().item() ** 3
    for t in range(1, max_it + 1):
        A = S @ Q - I
        gr = S @ A @ S
        if method == "gd":
            Q = Q - gr / Lmax
        elif method == "signgd":
            Q = Q - eta0 * rho ** t * torch.sign(gr)
        else:  # simplified Muon: exact polar factor, no momentum
            Uu, _, Vh = torch.linalg.svd(gr)
            Q = Q - eta0 * rho ** t * (Uu @ Vh)
        A = S @ Q - I
        f = 0.5 * torch.trace(A @ S @ A.T).item()
        if not math.isfinite(f) or f > 1e6 * f0:
            return None
        if f / f0 <= eps:
            return t
    return None


def ma_sweep(d=32, kappas=(1, 2, 4, 8, 16), eps=1e-6):
    rows = []
    common = [0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0]  # same step-size grid for both methods
    grids = {"muon": common, "signgd": common}
    rhos = [0.9, 0.95, 0.98, 0.99, 0.995]
    for rotated in (False, True):
        for kap in kappas:
            S, smin = ma_problem(d, kap, rotated)
            row = dict(kappa=kap, rotated=rotated, kappa_eff=kap ** 3)
            row["gd"] = ma_run(S, "gd", eps=eps, max_it=200_000)
            for m, Cs in grids.items():
                best = None
                for C in Cs:
                    for rho in rhos:
                        it = ma_run(S, m, eta0=C / smin, rho=rho, eps=eps)
                        if it is not None and (best is None or it < best[0]):
                            best = (it, C, rho)
                row[m] = best[0] if best else None
                row[m + "_cfg"] = best[1:] if best else None
            rows.append(row)
            print(row, flush=True)
    return rows


# ----------------------------------------------------------------------------- deep linear
def deeplin(method, s=(4.0, 2.0, 1.0, 0.5, 0.25, 0.125), rotated=True, init=1e-3, lr=None, steps=20000, seed=0):
    d = len(s)
    g = torch.Generator().manual_seed(seed)
    Uu = torch.tensor(haar_rotation(d, 11) if rotated else np.eye(d))
    Vv = torch.tensor(haar_rotation(d, 12) if rotated else np.eye(d))
    A = Uu @ torch.diag(torch.tensor(s)) @ Vv.T
    W1 = (init * torch.randn(d, d, generator=g)).requires_grad_()
    W2 = (init * torch.randn(d, d, generator=g)).requires_grad_()
    if method == "adam":
        opt = torch.optim.Adam([W1, W2], lr=lr)
    hist_t, hist_m = [], []
    for t in range(steps + 1):
        P = W2 @ W1
        if t % 10 == 0:
            m = torch.diagonal(Uu.T @ P.detach() @ Vv) / torch.tensor(s)
            hist_t.append(t)
            hist_m.append(m.tolist())
        L = 0.5 * ((P - A) ** 2).sum()
        g1, g2 = torch.autograd.grad(L, [W1, W2])
        with torch.no_grad():
            if method == "gd":
                W1 -= lr * g1
                W2 -= lr * g2
            elif method == "spectral":
                for W, gg in ((W1, g1), (W2, g2)):
                    u, _, vh = torch.linalg.svd(gg)
                    W -= lr * (u @ vh)
            else:
                W1.grad, W2.grad = g1, g2
                opt.step()
    M = np.array(hist_m)
    T90 = [next((hist_t[j] for j in range(len(hist_t)) if M[j, i] > 0.9), None) for i in range(d)]
    return dict(s=list(s), t=hist_t, m=M.tolist(), T90=T90, method=method, rotated=rotated, lr=lr)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--parts", default="zhang,ma,deeplin")
    a = ap.parse_args()
    torch.set_num_threads(1)
    res = {}
    if "zhang" in a.parts:
        res["zhang"] = {}
        for name, alpha, tmax in (("1/i", 1.0, 2e4), ("1/i2", 2.0, 2e5)):
            lam = spectrum(8, alpha)
            U = haar_rotation(8, 1234)
            for mode, init in (("sep", 5e-3), ("merged", 1e-3)):
                r = zhang_flow(lam, U, mode, init=init, t_max=tmax)
                r["plateaus"] = plateau_check(r)
                res["zhang"][f"{name}|{mode}"] = r
                print(f"zhang {name} {mode}: wall {r['wall']}s, final loss {r['loss'][-1]:.4f} vs optimum "
                      f"{r['L_star']:.4f}, |G-G*|/|G*| = {r['final_rel_err_to_Gstar']:.3g}", flush=True)
                for row in r["plateaus"]:
                    print("    ", row, flush=True)
        lam = np.ones(8)
        r = zhang_flow(lam, haar_rotation(8, 1234), "merged", init=1e-3, t_max=1e3)
        res["zhang"]["iso|merged"] = r
        print(f"zhang iso merged (von Oswald check): final loss {r['loss'][-1]:.4f} vs {r['L_star']:.4f}, "
              f"|G-G*|/|G*| = {r['final_rel_err_to_Gstar']:.3g}", flush=True)
        json.dump(res, open(a.out, "w"))
    if "ma" in a.parts:
        res["ma"] = ma_sweep()
        json.dump(res, open(a.out, "w"))
    if "deeplin" in a.parts:
        res["deeplin"] = []
        for rotated in (False, True):
            for method, lrs in (("gd", [0.02, 0.05, 0.1]), ("adam", [1e-3, 3e-3, 1e-2]), ("spectral", [3e-3, 1e-2, 3e-2])):
                runs = [deeplin(method, rotated=rotated, lr=lr) for lr in lrs]
                for r in runs:
                    print(f"deeplin {method} rotated={rotated} lr={r['lr']}: T90 per mode {r['T90']}", flush=True)
                res["deeplin"].extend(runs)
        json.dump(res, open(a.out, "w"))
    print("wrote", a.out)


if __name__ == "__main__":
    main()
