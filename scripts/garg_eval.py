"""Garg et al. (2022)-style evaluation of trained models.

For each prompt setting, error vs number of in-context examples k for the model(s) and Garg's
baselines (least squares, 3-nearest-neighbours, averaging), normalised by the setting's mean
squared target so that predicting 0 scores 1.  Settings (d = 8, 33 points, fresh w ~ N(0, I)):

  standard       x ~ N(0, I)                      (the training distribution of iso models)
  skewed         x ~ N(0, S), eig(S) ∝ 1/i^2, random rotation, trace d (context AND query)
  noisy          standard x, y + N(0, 1) noise (target includes the noise)
  scale_x2 / 0.5 standard x scaled by 2 / 0.5
  half_subspace  x confined to a random 4-dim subspace (trace d)
  orthogonal     query projected off the span of the k context points, rescaled (k < d)
  overlap        query = a random earlier context point

  python scripts/garg_eval.py out.json runA.json [runB.json ...]
"""
from __future__ import annotations

import json
import math
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from icl.data import haar_rotation  # noqa: E402
from icl.model import GPT2ICL  # noqa: E402
from icl.train import predict  # noqa: E402

D, NPTS, P = 8, 33, 1000
torch.set_num_threads(1)


def load_model(run_json):
    c = json.load(open(run_json))["config"]
    pt, ck = run_json.replace(".json", ".pt"), run_json.replace(".json", ".ckpt")
    sd = torch.load(pt) if os.path.exists(pt) else torch.load(ck, weights_only=False)["model"]
    m = GPT2ICL(c["d"], c["n_points"], c["n_embd"], c["n_layer"], c["n_head"], c["y_mode"], c.get("readin_init", "torch"))
    m.load_state_dict(sd)
    m.eval()
    return m, c


def ls_pred(X, Y, xq):
    w = torch.linalg.pinv(X) @ Y.unsqueeze(-1)
    return (w.squeeze(-1) * xq).sum(-1)


def knn_pred(X, Y, xq, k=3):
    dist = ((X - xq.unsqueeze(1)) ** 2).sum(-1)
    kk = min(k, X.shape[1])
    idx = dist.topk(kk, largest=False).indices
    return torch.gather(Y, 1, idx).mean(1)


def avg_pred(X, Y, xq):
    w = (X * Y.unsqueeze(-1)).mean(1)
    return (w * xq).sum(-1)


def ridge_pred(lam):
    def f(X, Y, xq):
        G = X.mT @ X + lam * torch.eye(X.shape[-1], dtype=X.dtype)
        w = torch.linalg.solve(G, X.mT @ Y.unsqueeze(-1))
        return (w.squeeze(-1) * xq).sum(-1)
    return f


BASELINES = {"least squares": ls_pred, "3-NN": knn_pred, "averaging": avg_pred}


def prompts(setting, g):
    x = torch.randn(P, NPTS, D, generator=g, dtype=torch.float64)
    w = torch.randn(P, D, generator=g, dtype=torch.float64)
    noise = None
    if setting == "skewed":
        lam = np.arange(1, D + 1) ** -2.0
        lam = lam * D / lam.sum()
        A = torch.tensor(haar_rotation(D, 4321) @ np.diag(np.sqrt(lam)))
        x = x @ A.T
    elif setting == "scale_x2":
        x = 2 * x
    elif setting == "scale_x0.5":
        x = 0.5 * x
    elif setting == "half_subspace":
        B = torch.tensor(haar_rotation(D, 98765)[:, : D // 2])
        x = (torch.randn(P, NPTS, D // 2, generator=g, dtype=torch.float64) * math.sqrt(2.0)) @ B.T
    y = torch.einsum("pnd,pd->pn", x, w)
    if setting == "noisy":
        noise = torch.randn(P, NPTS, generator=g, dtype=torch.float64)
        y = y + noise
    return x, y


def full_prompt_curves(models, setting, seed):
    g = torch.Generator().manual_seed(seed)
    x, y = prompts(setting, g)
    norm = (y[:, 1:] ** 2).mean().item()
    out = {}
    for name, m in models.items():
        pr = predict(m, x.float(), y.float()).double()
        e = ((pr - y) ** 2)
        out[name] = (e.mean(0) / norm).tolist()
        # bootstrap 95% CI of the k=1..32 mean error
        bs = [e[torch.randint(0, P, (P,), generator=g)][:, 1:].mean().item() / norm for _ in range(200)]
        out[name + " [CI k1-32]"] = [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]
    bl = dict(BASELINES)
    if setting == "noisy":
        bl["ridge (Bayes, λ=1)"] = ridge_pred(1.0)
    for bname, f in bl.items():
        pr = torch.zeros_like(y)
        for k in range(1, NPTS):
            pr[:, k] = f(x[:, :k], y[:, :k], x[:, k])
        out[bname] = (((pr - y) ** 2).mean(0) / norm).tolist()
    return out


def per_k_curves(models, setting, seed):
    """orthogonal / overlap queries: a separate forward pass per prefix length k (Garg's eval_batch)."""
    g = torch.Generator().manual_seed(seed)
    x, y = prompts("standard", g)
    w = torch.linalg.lstsq(x[:, :D], y[:, :D].unsqueeze(-1)).solution.squeeze(-1)  # exact w (noiseless)
    ks = range(1, D) if setting == "orthogonal" else range(1, NPTS)
    res = {n: [None] * NPTS for n in list(models) + list(BASELINES)}
    for k in ks:
        xq = x[:, k].clone()
        if setting == "orthogonal":
            Qb, _ = torch.linalg.qr(x[:, :k].mT)  # (P, D, k) orthonormal basis of the context span
            proj = xq - (Qb @ (Qb.mT @ xq.unsqueeze(-1))).squeeze(-1)
            xq = proj * (xq.norm(dim=-1, keepdim=True) / proj.norm(dim=-1, keepdim=True).clamp_min(1e-12))
        else:
            j = torch.randint(0, k, (P,), generator=g)
            xq = x[torch.arange(P), j]
        yq = (w * xq).sum(-1)
        xx = torch.cat([x[:, :k], xq.unsqueeze(1)], 1)
        yy = torch.cat([y[:, :k], yq.unsqueeze(1)], 1)
        norm = (yq ** 2).mean().item()
        for name, m in models.items():
            pr = predict(m, xx.float(), yy.float())[:, -1].double()
            res[name][k] = ((pr - yq) ** 2).mean().item() / norm
        for bname, f in BASELINES.items():
            res[bname][k] = ((f(x[:, :k], y[:, :k], xq) - yq) ** 2).mean().item() / norm
    return res


SETTINGS = ["standard", "skewed", "noisy", "scale_x2", "scale_x0.5", "half_subspace"]


def main():
    out_path, runs = sys.argv[1], sys.argv[2:]
    models, configs = {}, {}
    for rj in runs:
        m, c = load_model(rj)
        name = os.path.basename(rj).replace(".json", "")
        models[name], configs[name] = m, c
    res = dict(models={n: dict(opt=c["opt"], lr=c["lr"], cond=c["cond"], steps=c["steps"], y_mode=c["y_mode"],
                               curriculum=c.get("curriculum", "none"), seed=c["seed"]) for n, c in configs.items()})
    for i, s in enumerate(SETTINGS):
        res[s] = full_prompt_curves(models, s, 50_000 + i)
    for i, s in enumerate(["orthogonal", "overlap"]):
        res[s] = per_k_curves(models, s, 60_000 + i)
    json.dump(res, open(out_path, "w"))
    print("wrote", out_path)
    for s in SETTINGS + ["orthogonal", "overlap"]:
        ks = [1, 4, 7, 8, 10, 16, 32] if s != "orthogonal" else [1, 3, 5, 7]
        print(f"\n[{s}]  k = " + " ".join(f"{k:>6d}" for k in ks))
        for n, v in res[s].items():
            if "[CI" in n:
                continue
            print(f"  {n[:34]:34s} " + " ".join(f"{(v[k] if v[k] is not None else float('nan')):6.3f}" for k in ks))


if __name__ == "__main__":
    main()
