"""Exploratory head alignment, accounting for the input embedding.

For layer l, fit linear maps from the raw inputs to that layer's LayerNorm output:
  J_q : x_t      -> ln_1(h)[x-token t]   (queries live on x tokens: the prediction positions)
  J_k : x_t      -> ln_1(h)[y-token t]   (keys/values that carry (x_t, y_t) live on y tokens)
by least squares over many prompts (R^2 reports how linear the map is).  Each head's effective
bilinear form in input space is then  B_h = J_q^T W_Q,h^T W_K,h J_k  (d x d).  In the whitened
principal-axis basis (Lam^1/2 U^T B_h U Lam^1/2, so that x_q^T Sigma^-1 x_k scores every axis
equally), the share of ||B_h||_F^2 on axis i's diagonal entry says which input directions head h
compares; `axis_share` is the norm-weighted average over heads.  Layer 1 sees the read-in directly; deeper
layers see the read-in composed with the earlier blocks, captured (to first order) by J.
"""
from __future__ import annotations

import numpy as np
import torch


@torch.no_grad()
def head_alignment(model, cond, n_prompts=256, seed=0):
    from .data import Sampler

    model.eval()
    s = Sampler(cond, n_points=model.wpe.weight.shape[0] // 2, seed=seed)
    x, y = s.sample(n_prompts)
    lns = []
    hooks = [blk.ln_1.register_forward_hook(lambda m, i, o: lns.append(o)) for blk in model.blocks]
    model(x, y)
    for h in hooks:
        h.remove()
    U = torch.tensor(cond.U, dtype=torch.float64)
    d = x.shape[-1]
    X = torch.cat([x.reshape(-1, d).double(), torch.ones(x.shape[0] * x.shape[1], 1, dtype=torch.float64)], 1)
    out = []
    for l, (blk, a) in enumerate(zip(model.blocks, lns)):
        a = a.double()
        aq = a[:, 0::2].reshape(-1, a.shape[-1])  # x tokens (n of them)
        ak = a[:, 1::2].reshape(-1, a.shape[-1])  # y tokens (n-1; the last y is dropped)
        Xk = X.reshape(x.shape[0], x.shape[1], -1)[:, : a[:, 1::2].shape[1]].reshape(-1, d + 1)
        Jq = torch.linalg.lstsq(X, aq).solution[:d].T  # (C, d)
        Jk = torch.linalg.lstsq(Xk, ak).solution[:d].T
        r2q = 1 - ((aq - X @ torch.linalg.lstsq(X, aq).solution) ** 2).sum() / ((aq - aq.mean(0)) ** 2).sum()
        r2k = 1 - ((ak - Xk @ torch.linalg.lstsq(Xk, ak).solution) ** 2).sum() / ((ak - ak.mean(0)) ** 2).sum()
        Wq, Wk = blk.q.weight.double(), blk.k.weight.double()
        H = blk.n_head
        hd = Wq.shape[0] // H
        heads = []
        for h in range(H):
            B = Jq.T @ Wq[h * hd:(h + 1) * hd].T @ Wk[h * hd:(h + 1) * hd] @ Jk  # (d, d) in input coords
            # principal-axis basis, whitened: a head computing x_q^T Sigma^-1 x_k uses every axis equally
            sl = torch.tensor(np.sqrt(cond.lam), dtype=torch.float64)
            Bp = sl[:, None] * (U.T @ B @ U) * sl[None, :]
            tot = (Bp ** 2).sum()
            heads.append(dict(diag_share=(torch.diagonal(Bp) ** 2 / tot).tolist(),
                              diag_total=float((torch.diagonal(Bp) ** 2).sum() / tot),
                              norm=float(tot.sqrt())))
        out.append(dict(layer=l, r2_q=float(r2q), r2_k=float(r2k), heads=heads))
    return out


def summarize(align):
    """Per layer: norm-weighted share of each principal axis on head diagonals (sums to diag_total)."""
    rows = []
    for L in align:
        w = np.array([h["norm"] ** 2 for h in L["heads"]])
        share = np.array([h["diag_share"] for h in L["heads"]])
        rows.append(dict(layer=L["layer"], r2_q=L["r2_q"], r2_k=L["r2_k"],
                         axis_share=(w[:, None] * share).sum(0) / w.sum(),
                         per_head_top_axis=[int(np.argmax(s)) + 1 for s in share],
                         per_head_diag_total=[h["diag_total"] for h in L["heads"]]))
    return rows
