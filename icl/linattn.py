"""One-layer linear attention (Zhang et al., 2025) on the same task, with d = 8 and N = 32 examples.

The model predicts y_q = h^T Gamma x_q, where h = (1/N) sum_n y_n x_n and Gamma = sum_r v_r k_r q_r^T over 8 heads
with rank-one keys and queries. Its expected loss is an exact function of Gamma, so the logged losses are exact.

  python -m icl.linattn runs/linattn.json

For each optimizer: a six-point learning-rate sweep on seed 0 (4,000 steps, the rate with the lowest mean loss is
kept), then seeds 0-2 at that rate for 6,000 steps, logged at every step.
"""
import argparse
import json
import math
from multiprocessing import Pool

import numpy as np
import torch

from .data import scale, spectrum
from .optim import Chain, Muon

D, N, HEADS, BATCH = 8, 32, 8, 1024
POWERS = (1, 0.75, 0.5, 0.25, 0.1, 0)
ARMS = ["sgd", "adam", "muon"] + [f"muon_p{p:g}" for p in POWERS]
MUON_LRS = [1e-3, 3e-3, 1e-2, 2e-2, 5e-2, 1e-1]
LRS = {"sgd": [3e-3, 1e-2, 2e-2, 5e-2, 0.1, 0.2], "adam": [3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 0.1]}


class LinearAttention(torch.nn.Module):
    def __init__(self, init=1e-2):
        super().__init__()
        self.K = torch.nn.Parameter(init * torch.randn(HEADS, D))
        self.Q = torch.nn.Parameter(init * torch.randn(HEADS, D))
        self.v = torch.nn.Parameter(init * torch.randn(HEADS))

    def gamma(self):
        return torch.einsum("a,ai,aj->ij", self.v, self.K, self.Q)

    def forward(self, X, Y, xq):
        h = (X * Y.unsqueeze(-1)).mean(1)
        return torch.einsum("bi,ij,bj->b", h, self.gamma(), xq)


def optimizer(model, arm, lr):
    """SGD here is plain gradient descent, as in Zhang et al.; Muon trains K and Q, Adam the head scalars v."""
    if arm == "sgd":
        return torch.optim.SGD(model.parameters(), lr=lr)
    if arm == "adam":
        return torch.optim.Adam(model.parameters(), lr=lr)
    power = None if arm == "muon" else float(arm[len("muon_p"):])
    return Chain(Muon([model.K, model.Q], lr, power=power), torch.optim.Adam([model.v], lr=lr))


def losses(G, lam, lam_q):
    """Exact expected loss / d of y_q = h^T G x_q, in distribution and with queries drawn from N(0, diag(lam_q))."""
    S, Sq = torch.diag(lam), torch.diag(lam_q)
    A = G @ S @ G.T
    L = torch.trace((N + 1) / N * S @ A @ S + torch.trace(A @ S) / N * S) - 2 * torch.trace(S @ G.T @ S) + torch.trace(S)
    A = G @ Sq @ G.T
    Lq = torch.trace((N + 1) / N * S @ A @ S + torch.trace(A @ S) / N * S) - 2 * torch.trace(S @ G @ Sq) + torch.trace(Sq)
    return float(L) / D, float(Lq) / D


def train(arm, lr, seed, steps, log_every=1):
    """Exact losses (in distribution, and with queries shifted fully toward rare directions, lambda^-1) and the gain
    lambda_j Gamma_jj on each direction, every log_every steps."""
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    lam = torch.tensor(spectrum(D), dtype=torch.float32)
    lam_q = lam ** -1 * float(lam.sum()) / (lam ** -1).sum()
    model = LinearAttention()
    opt = optimizer(model, arm, lr)
    A = scale(spectrum(D))
    g = torch.Generator().manual_seed(seed + 1)
    out = dict(arm=arm, lr=lr, seed=seed, loss=[], qloss=[], gain=[])
    for step in range(steps + 1):
        if step % log_every == 0:
            with torch.no_grad():
                G = model.gamma()
                L, Lq = losses(G, lam, lam_q)
            if not math.isfinite(L) or L > 1e6:
                return dict(out, diverged=True)
            out["loss"].append(L)
            out["qloss"].append(Lq)
            out["gain"].append((lam * torch.diagonal(G)).tolist())
        if step == steps:
            return out
        x = torch.randn(BATCH, N + 1, D, generator=g) @ A
        w = torch.randn(BATCH, D, generator=g)
        y = torch.einsum("bnd,bd->bn", x, w)
        loss = ((model(x[:, :N], y[:, :N], x[:, N]) - y[:, N]) ** 2).mean()
        for p in model.parameters():
            p.grad = None
        loss.backward()
        opt.step()


def sweep_score(job):
    out = train(*job, steps=4000, log_every=10)
    return np.inf if out.get("diverged") else float(np.mean(out["loss"]))


def final_run(job):
    out = train(*job, steps=6000)
    for k in ("loss", "qloss", "gain"):
        out[k] = np.vectorize(lambda v: float(f"{v:.6g}"))(np.array(out[k])).tolist()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--procs", type=int, default=4)
    a = ap.parse_args()
    sweep = [(arm, lr, 0) for arm in ARMS for lr in LRS.get(arm, MUON_LRS)]
    with Pool(a.procs) as pool:
        scores = pool.map(sweep_score, sweep, chunksize=1)
        best = {}
        for (arm, lr, _), s in zip(sweep, scores):
            if arm not in best or s < best[arm][1]:
                best[arm] = (lr, s)
        runs = pool.map(final_run, [(arm, best[arm][0], seed) for arm in ARMS for seed in (0, 1, 2)], chunksize=1)
    with open(a.out, "w") as f:
        json.dump(dict(lam=spectrum(D).tolist(), n=N, runs=runs), f)


if __name__ == "__main__":
    main()
