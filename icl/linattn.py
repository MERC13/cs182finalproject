"""Theory testbed: one-layer multi-head linear attention (Zhang et al. 2025 / Ma et al. 2026).

Prediction for query x_q from context (x_n, y_n), n = 1..N, with beta = (1/N) sum_n y_n x_n:
  merged  (ATTN_M):        yhat = beta^T (sum_h v_h U_h) x_q              U_h in R^{d x d}
  separate (ATTN_S, rank r): yhat = sum_h v_h beta^T K_h^T Q_h x_q          K_h, Q_h in R^{r x d}
Both are yhat = beta^T Gamma x_q.  In the principal-axis basis the model's gain on axis i is
rho_i = lam_i (U^T Gamma U)_ii; the finite-N optimum (one-step preconditioned GD) has
rho_i* = lam_i N / ((N+1) lam_i + tr Lam)  (Zhang et al. eq. 22).

Optimizers: "gd" (plain SGD, no momentum, as in Zhang's simulations), "adam", "muon"
(Newton-Schulz, or "muon_exact" with the SVD polar factor, on the stacked U / K / Q matrices;
the per-head scalars v use Adam), "signgd", and "muon_pow<p>" (Muon with the update's singular
values raised to p: 1 = normalised gradient, 0 = exact polar factor).
"""
from __future__ import annotations

import argparse
import json
import math
import time

import numpy as np
import torch

from .data import make_condition
from .optim import Muon


class LinAttn(torch.nn.Module):
    def __init__(self, d, H, mode="sep", rank=1, init=1e-2):
        super().__init__()
        self.mode = mode
        g = lambda *s: torch.nn.Parameter(init * torch.randn(*s))
        if mode == "merged":
            self.U = g(H, d, d)
        else:
            self.K = g(H * rank, d)
            self.Q = g(H * rank, d)
            self.H, self.r = H, rank
        self.v = g(H)

    def gamma(self):
        if self.mode == "merged":
            return torch.einsum("h,hij->ij", self.v, self.U)
        v = self.v.repeat_interleave(self.r)
        return torch.einsum("a,ai,aj->ij", v, self.K, self.Q)

    def forward(self, X, Y, xq):
        beta = (X * Y.unsqueeze(-1)).mean(1)
        return torch.einsum("bi,ij,bj->b", beta, self.gamma(), xq)

    def matrices(self):
        return [self.U] if self.mode == "merged" else [self.K, self.Q]


def run(cond_name, opt, lr, mode="sep", H=8, rank=1, d=8, N=32, steps=4000, batch=1024, init=1e-2,
        seed=0, log_every=20, aux_lr=None, muon_momentum=0.95, ema=0.0, log_gamma=False):
    torch.manual_seed(seed)
    cond = make_condition(cond_name, d)
    lam = torch.tensor(cond.lam, dtype=torch.float32)
    U = torch.tensor(cond.U, dtype=torch.float32)
    A = cond.A()
    m = LinAttn(d, H, mode, rank, init)
    if opt == "gd":
        o = [torch.optim.SGD(m.parameters(), lr=lr)]
    elif opt == "signgd":
        o = None
    elif opt == "adam":
        o = [torch.optim.Adam(m.parameters(), lr=lr)]
    elif opt in ("muon", "muon_exact"):
        o = [Muon(m.matrices(), lr=lr, momentum=muon_momentum, orth="ns" if opt == "muon" else "exact"),
             torch.optim.Adam([m.v], lr=aux_lr or lr)]
    elif opt.startswith("muon_pow"):  # "muon_pow0.5": singular values of the update raised to 0.5
        o = [Muon(m.matrices(), lr=lr, momentum=muon_momentum, orth="pow", power=float(opt[len("muon_pow"):])),
             torch.optim.Adam([m.v], lr=aux_lr or lr)]
    else:
        raise ValueError(opt)
    g = torch.Generator().manual_seed(seed + 1)
    T = float(lam.sum())
    rho_star = lam * N / ((N + 1) * lam + T)
    out = dict(cond=cond_name, opt=opt, lr=lr, mode=mode, H=H, rank=rank, init=init, seed=seed,
               muon_momentum=muon_momentum if opt.startswith("muon") else None,
               lam=cond.lam.tolist(), rho_star=rho_star.tolist(), steps=[], loss=[], rho=[], qloss=[])
    # inverted query spectrum at the same trace (query-shift test, t = 1)
    lq = torch.tensor(cond.lam[::-1].copy(), dtype=torch.float32)
    lq = (lam ** -1) * T / (lam ** -1).sum()
    Sq = U @ torch.diag(lq) @ U.T
    S = U @ torch.diag(lam) @ U.T

    def pop(Gm):
        # exact population losses of yhat = beta^T Gamma x_q (Gaussian identity), normalised by d
        rho = lam * torch.diagonal(U.T @ Gm @ U)
        Ag = Gm @ S @ Gm.T  # Gamma Sigma Gamma^T
        ESAS = (N + 1) / N * S @ Ag @ S + torch.trace(Ag @ S) / N * S
        L = torch.trace(ESAS) - 2 * torch.trace(S @ Gm.T @ S) + torch.trace(S)
        Aq = Gm @ Sq @ Gm.T
        ESq = (N + 1) / N * S @ Aq @ S + torch.trace(Aq @ S) / N * S
        Lq = torch.trace(ESq) - 2 * torch.trace(S @ Gm @ Sq) + torch.trace(Sq)
        return float(L) / d, float(Lq) / d, rho.tolist()

    # ema > 0: also log a running average of Gamma (= an average of predictors, since yhat is linear in
    # Gamma), which removes step-to-step parameter jitter but keeps the learning path
    G_ema = m.gamma().detach().clone() if ema else None
    if ema:
        out.update(loss_ema=[], qloss_ema=[], rho_ema=[], ema=ema)
    for step in range(steps + 1):
        if step % log_every == 0:
            with torch.no_grad():
                L, Lq, rho = pop(m.gamma())
                if ema:
                    Le, Lqe, rhoe = pop(G_ema)
                    out["loss_ema"].append(Le); out["qloss_ema"].append(Lqe); out["rho_ema"].append(rhoe)
            out["qloss"].append(Lq)
            if log_gamma:  # the full learned matrix (principal basis), for exact losses under any test shift
                out.setdefault("gamma", []).append((U.T @ m.gamma().detach() @ U).tolist())
            out["steps"].append(step)
            out["loss"].append(L)
            out["rho"].append(rho)
            if not math.isfinite(L) or L > 1e6:
                out["status"] = f"diverged@{step}"
                return out
        if step == steps:
            break
        z = torch.randn(batch, N + 1, d, generator=g)
        x = z @ A.T
        w = torch.randn(batch, d, generator=g)
        y = torch.einsum("bnd,bd->bn", x, w)
        loss = ((m(x[:, :N], y[:, :N], x[:, N]) - y[:, N]) ** 2).mean()
        for p in m.parameters():
            p.grad = None
        loss.backward()
        if opt == "signgd":
            with torch.no_grad():
                for p in m.parameters():
                    p -= lr * p.grad.sign()
        else:
            for oo in o:
                oo.step()
        if ema:
            with torch.no_grad():
                G_ema.mul_(ema).add_(m.gamma(), alpha=1 - ema)
    out["status"] = "done"
    return out


LR_GRID = {"gd": [0.01, 0.02, 0.05, 0.1, 0.2], "signgd": [3e-4, 1e-3, 3e-3, 1e-2, 3e-2],
           "adam": [1e-3, 3e-3, 1e-2, 3e-2, 1e-1], "muon": [3e-3, 1e-2, 2e-2, 5e-2, 1e-1],
           "muon_exact": [3e-3, 1e-2, 2e-2, 5e-2, 1e-1]}
CONDS = ["iso", "1/i-ali", "1/i-rot", "1/i2-ali", "1/i2-rot"]


def score(r):
    """Selection metric: mean loss over training (area under the curve; rewards speed and final value)."""
    return float(np.mean(r["loss"])) if r["status"] == "done" else float("inf")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--modes", default="sep,merged")
    a = ap.parse_args()
    torch.set_num_threads(1)
    t0 = time.time()
    res = []
    for mode in a.modes.split(","):
        for cond in CONDS:
            for opt, grid in LR_GRID.items():
                sweep = [run(cond, opt, lr, mode=mode, steps=a.steps, log_every=10) for lr in grid]
                best = min(sweep, key=score)
                best["selected"] = True
                for r in sweep:
                    r["sweep"] = True
                res.extend(sweep)
                for seed in range(1, a.seeds):
                    r = run(cond, opt, best["lr"], mode=mode, steps=a.steps, log_every=10, seed=seed)
                    r["selected"] = True
                    res.append(r)
                print(f"[{time.time()-t0:6.0f}s] {mode} {cond} {opt}: best lr {best['lr']} "
                      f"final loss {best['loss'][-1]:.4f}", flush=True)
                json.dump(res, open(a.out, "w"))
    print("done in", time.time() - t0)


# ----------------------------------------------------------------------------- exact-gradient robustness sweep
def run_exact(cond_name, opt, lr, mode="sep", H=8, d=8, N=32, steps=4000, init=1e-2, seed=0, log_every=10):
    """Same model and optimizers as `run`, but trained on the exact population loss (no sampling noise)."""
    from .repro_theory import pop_loss
    torch.manual_seed(seed)
    cond = make_condition(cond_name, d)
    lam = torch.tensor(cond.lam, dtype=torch.float64)
    U = torch.tensor(cond.U, dtype=torch.float64)
    S = U @ torch.diag(lam) @ U.T
    m = LinAttn(d, H, mode, 1, init).double()
    if opt == "gd":
        o = [torch.optim.SGD(m.parameters(), lr=lr)]
    elif opt == "adam":
        o = [torch.optim.Adam(m.parameters(), lr=lr)]
    elif opt in ("muon", "muon_exact"):
        o = [Muon(m.matrices(), lr=lr, orth="ns" if opt == "muon" else "exact"), torch.optim.Adam([m.v], lr=lr)]
    else:
        o = None
    T = float(lam.sum())
    rho_star = lam * N / ((N + 1) * lam + T)
    Lstar = float(pop_loss(U @ torch.diag(N / ((N + 1) * lam + T)) @ U.T, S, N))
    steps_l, loss_l, rho_l = [], [], []
    for step in range(steps + 1):
        L = pop_loss(m.gamma(), S, N)
        if step % log_every == 0:
            with torch.no_grad():
                G = U.T @ m.gamma() @ U
                steps_l.append(step); loss_l.append(float(L)); rho_l.append((lam * torch.diagonal(G) / rho_star).tolist())
        if not math.isfinite(float(L.detach())) or float(L.detach()) > 1e6:
            return dict(status="diverged")
        if step == steps:
            break
        for p in m.parameters():
            p.grad = None
        L.backward()
        if opt == "signgd":
            with torch.no_grad():
                for p in m.parameters():
                    p -= lr * p.grad.sign()
        else:
            for oo in o:
                oo.step()
    st = np.array(steps_l, float); Lr = np.array(loss_l); R = np.array(rho_l)
    exc = (Lr - Lstar) / (1 - Lstar)
    first = lambda mask: float(st[np.nonzero(mask)[0][0]]) if mask.any() else float("nan")
    return dict(status="done", T_conv=first(exc < 0.05), T_axis=[first(R[:, i] > 0.9) for i in range(R.shape[1])],
                final_exc=float(exc[-1]), auc=float(np.mean(np.minimum(exc, 1.0))))


def exact_sweep(out_path, steps=3000):
    lrs = {"gd": np.logspace(-2.5, 0, 8), "signgd": np.logspace(-4, -1, 8), "adam": np.logspace(-3.5, -0.5, 8),
           "muon": np.logspace(-3, -0.5, 8), "muon_exact": np.logspace(-3, -0.5, 8)}
    rows = []
    t0 = time.time()
    for mode in ("sep", "merged"):
        for init in (1e-3, 1e-2):
            for cond in ("iso", "1/i2-ali", "1/i2-rot"):
                for opt, grid in lrs.items():
                    for lr in grid:
                        r = run_exact(cond, opt, float(lr), mode=mode, init=init, steps=steps)
                        r.update(mode=mode, init=init, cond=cond, opt=opt, lr=float(lr))
                        rows.append(r)
                print(f"[{time.time()-t0:5.0f}s] {mode} init={init} done", flush=True)
                json.dump(rows, open(out_path, "w"))
    return rows
