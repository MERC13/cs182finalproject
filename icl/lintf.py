"""Multi-layer linear transformers for in-context regression: the rungs between the one-layer testbed and the GPT-2.

Tokens are concatenated, z_i = [x_i, y_i] for the N context examples and [x_q, 0] for the query (the layout
of von Oswald et al. 2023 and Ahn et al. 2023). Each layer adds linear attention over the context tokens,

    Z <- Z + (1/N) sum_h O_h V_h C (K_h C)^T Q_h Z,      C = the current context tokens,

with separate per-head Q, K, V (dh x D) and O (D x dh), as in the GPT-2's blocks but without softmax, MLP or
LayerNorm. Two read-in arms:
  readin = "none"    the layers act on the raw (d+1)-dim tokens (D = d + 1), so rotating x reaches both sides
                     of every weight matrix; the prediction is the query token's y slot after the last layer.
  readin = "learned" a Gaussian Linear(d+1, D) read-in into a wider stream and a learned Linear(D, 1) read-out,
                     as in the GPT-2 (both trained by AdamW in the Muon arm, as there).
Per-axis learning is read from probe prompts whose task lies along one eigenvector of the input covariance:
axis j counts as learned once the probe error, normalised by that axis's output variance, drops below 0.5.
"""
from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn

from .data import make_condition
from .optim import Muon


class LinTF(nn.Module):
    def __init__(self, d: int, L: int = 2, H: int = 4, readin: str = "none", D: int = 64, init: float = 0.1):
        super().__init__()
        fan = d + 1
        self.readin = readin
        self.D = fan if readin == "none" else D
        dh = self.D if readin == "none" else self.D // H
        s = init / math.sqrt(self.D)
        mk = lambda *shape: nn.Parameter(s * torch.randn(*shape))
        self.Q = nn.ParameterList([mk(H, dh, self.D) for _ in range(L)])
        self.K = nn.ParameterList([mk(H, dh, self.D) for _ in range(L)])
        self.V = nn.ParameterList([mk(H, dh, self.D) for _ in range(L)])
        self.O = nn.ParameterList([mk(H, self.D, dh) for _ in range(L)])
        if readin == "learned":
            self.E = nn.Linear(fan, self.D)
            nn.init.normal_(self.E.weight, std=1.0 / math.sqrt(3 * fan))
            nn.init.zeros_(self.E.bias)
            self.u = nn.Linear(self.D, 1)

    def forward(self, x, y, xq):
        B, N, d = x.shape
        Z = torch.cat([torch.cat([x, y.unsqueeze(-1)], -1),
                       torch.cat([xq, xq.new_zeros(B, 1)], -1).unsqueeze(1)], 1)  # (B, N+1, d+1)
        if self.readin == "learned":
            Z = self.E(Z)
        for Q, K, V, O in zip(self.Q, self.K, self.V, self.O):
            C = Z[:, :N]
            q = torch.einsum("hed,bnd->bhne", Q, Z)
            k = torch.einsum("hed,bnd->bhne", K, C)
            v = torch.einsum("hed,bnd->bhne", V, C)
            att = torch.einsum("bhqe,bhke->bhqk", q, k) / N
            Z = Z + torch.einsum("hde,bhqe->bqd", O, torch.einsum("bhqk,bhke->bhqe", att, v))
        zq = Z[:, -1]
        return zq[:, -1] if self.readin == "none" else self.u(zq).squeeze(-1)

    def matrices(self):
        return [*self.Q, *self.K, *self.V, *self.O]


def run(cond_name, opt, lr, L=2, readin="none", H=4, D=64, d=8, N=32, steps=4000, batch=256, init=0.1, seed=0,
        log_every=50, n_eval=2048, aux_lr=1e-3, momentum=0.9, clip=1.0):
    torch.manual_seed(seed)
    cond = make_condition(cond_name, d)
    A = cond.A()
    U = torch.tensor(cond.U, dtype=torch.float32)
    lam = np.asarray(cond.lam)
    m = LinTF(d, L, H, readin, D, init)
    mats = m.matrices()
    rest = [p for n, p in m.named_parameters() if n.startswith(("E.", "u."))]
    if opt == "sgd":
        o = [torch.optim.SGD(m.parameters(), lr=lr, momentum=momentum)]
    elif opt == "adam":
        o = [torch.optim.Adam(m.parameters(), lr=lr)]
    elif opt == "muon":
        o = [Muon(mats, lr=lr)] + ([torch.optim.Adam(rest, lr=aux_lr)] if rest else [])
    else:
        raise ValueError(opt)
    g = torch.Generator().manual_seed(seed + 1)
    ge = torch.Generator().manual_seed(12345)  # the same evaluation prompts for every run

    def prompts(n, gen, w=None):
        x = torch.randn(n, N + 1, d, generator=gen) @ A.T
        w = torch.randn(n, d, generator=gen) if w is None else w
        return x, torch.einsum("bnd,bd->bn", x, w)

    ev = prompts(n_eval, ge)
    probes = []  # task along eigenvector j (scaled N(0,1)), normalised by that axis's output variance lam_j
    for j in range(d):
        gj = torch.randn(n_eval // 4, 1, generator=ge)
        xp, yp = prompts(n_eval // 4, ge, w=gj * U[:, j])
        probes.append((xp, yp))
    out = dict(cond=cond_name, opt=opt, lr=lr, L=L, readin=readin, H=H, D=m.D, init=init, seed=seed, batch=batch,
               lam=lam.tolist(), steps=[], loss=[], probe=[])
    for step in range(steps + 1):
        if step % log_every == 0:
            with torch.no_grad():
                x, y = ev
                Lv = float(((m(x[:, :N], y[:, :N], x[:, N]) - y[:, N]) ** 2).mean()) / d
                pe = [float(((m(xp[:, :N], yp[:, :N], xp[:, N]) - yp[:, N]) ** 2).mean()) / lam[j]
                      for j, (xp, yp) in enumerate(probes)]
            out["steps"].append(step); out["loss"].append(Lv); out["probe"].append(pe)
            if not math.isfinite(Lv) or Lv > 1e6:
                out["status"] = f"diverged@{step}"
                return out
        if step == steps:
            break
        x, y = prompts(batch, g)
        loss = ((m(x[:, :N], y[:, :N], x[:, N]) - y[:, N]) ** 2).mean()
        for p in m.parameters():
            p.grad = None
        loss.backward()
        if opt == "sgd" and clip:
            torch.nn.utils.clip_grad_norm_(m.parameters(), clip)
        for oo in o:
            oo.step()
    out["status"] = "done"
    return out


def axis_times(r, thr=0.5):
    st, P = np.array(r["steps"], float), np.array(r["probe"])
    T = []
    for j in range(P.shape[1]):
        i = np.nonzero(P[:, j] < thr)[0]
        T.append(float(st[i[0]]) if len(i) else np.inf)
    return np.array(T)
