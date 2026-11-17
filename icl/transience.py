"""Finite-task-pool evaluation: does in-context learning persist or give way to memorisation?

Training tasks come from a fixed pool w_1..w_M (Raventos et al. 2023), with label noise sigma.
Two Bayes-optimal reference predictors bracket what the model can do:
  ridge  : the posterior mean under the true prior w ~ N(0, I) -> generalises to unseen tasks
           (in-context learning proper);
  dMMSE  : the posterior mean under the discrete prior "w is one of the M pool tasks" -> best on
           the pool, poor on unseen tasks (memorisation / in-weights task identification).
A transient-ridge run first moves towards ridge and later towards dMMSE (Carroll et al. 2025).

Evaluation prompts have noisy context labels (as in training); errors are measured against the
noiseless target w.x, normalised by d, so the zero predictor scores ~1 on isotropic inputs.
"""
from __future__ import annotations

import torch


def ridge_prefix(x: torch.Tensor, y: torch.Tensor, lam: float) -> torch.Tensor:
    """Ridge (= Bayes posterior mean for w ~ N(0, I), noise var lam) prediction of y_i from pairs < i."""
    x, y = x.double(), y.double()
    P, n, d = x.shape
    xx = torch.einsum("pni,pnj->pnij", x, x).cumsum(1)
    xy = (x * y.unsqueeze(-1)).cumsum(1)
    S = torch.cat([x.new_zeros(P, 1, d, d), xx[:, :-1]], 1) + lam * torch.eye(d, dtype=x.dtype)
    b = torch.cat([x.new_zeros(P, 1, d), xy[:, :-1]], 1)
    w = torch.linalg.solve(S, b.unsqueeze(-1)).squeeze(-1)
    return (w * x).sum(-1).float()


def dmmse_prefix(x: torch.Tensor, y: torch.Tensor, pool: torch.Tensor, sigma2: float,
                 chunk: int = 64, pool_p: torch.Tensor | None = None) -> torch.Tensor:
    """Posterior-mean prediction under the training prior over the pool tasks (uniform unless pool_p
    gives task frequencies), noise var sigma2."""
    out = []
    W = pool.double()
    logp = 0.0 if pool_p is None else torch.log(pool_p.double())
    for i in range(0, len(x), chunk):
        xc, yc = x[i:i + chunk].double(), y[i:i + chunk].double()
        F = torch.einsum("pnd,md->pnm", xc, W)  # each pool task's prediction at each position
        R = (yc.unsqueeze(-1) - F) ** 2
        L = -torch.cat([torch.zeros_like(R[:, :1]), R[:, :-1].cumsum(1)], 1) / (2 * sigma2) + logp
        out.append((torch.softmax(L, -1) * F).sum(-1).float())
    return torch.cat(out)


def make_tr_sets(cond, n_points: int, seed: int, pool: torch.Tensor, noise: float, n: int = 512,
                 pool_p: torch.Tensor | None = None):
    """Prompts on unseen tasks ("fresh", w ~ N(0, I)) and on pool tasks ("pool"), with references.

    With task frequencies pool_p, "pool" follows the training frequencies, and two more sets hold the
    most frequent eighth of the tasks ("head") and the least frequent half ("tail"), each sampled
    uniformly within the group, so memorisation can be compared across frequencies."""
    g = torch.Generator().manual_seed(seed)
    A = cond.A()
    d = cond.d
    M = len(pool)
    groups = {"fresh": None, "pool": None}
    if pool_p is not None:
        groups.update(head=torch.arange(0, max(1, M // 8)), tail=torch.arange(M // 2, M))
    sets = {}
    for name, grp in groups.items():
        x = torch.randn(n, n_points, d, generator=g) @ A.T
        if name == "fresh":
            w = torch.randn(n, d, generator=g)
        elif name == "pool":
            idx = (torch.randint(M, (n,), generator=g) if pool_p is None
                   else torch.multinomial(pool_p, n, replacement=True, generator=g))
            w = pool[idx]
        else:
            w = pool[grp[torch.randint(len(grp), (n,), generator=g)]]
        f = torch.einsum("pnd,pd->pn", x, w)
        y = f + noise * torch.randn(f.shape, generator=g)
        sets[name] = dict(x=x, y=y, f=f, ridge=ridge_prefix(x, y, noise ** 2),
                          dmmse=dmmse_prefix(x, y, pool, noise ** 2, pool_p=pool_p))
    return sets


def reference_errors(sets, d: int):
    """Per-position error of each reference predictor on each set (fixed for the run)."""
    return {f"{s}_{ref}": (((v[ref] - v["f"]) ** 2).mean(0) / d).tolist()
            for s, v in sets.items() for ref in ("ridge", "dmmse")}


@torch.no_grad()
def tr_eval(predict, model, sets, d: int):
    """Model error and distance to each reference, per position, on fresh and pool tasks."""
    out = {}
    for s, v in sets.items():
        p = predict(model, v["x"], v["y"])
        out[f"tr_{s}_err_k"] = (((p - v["f"]) ** 2).mean(0) / d).tolist()
        out[f"tr_{s}_dridge_k"] = (((p - v["ridge"]) ** 2).mean(0) / d).tolist()
        out[f"tr_{s}_ddmmse_k"] = (((p - v["dmmse"]) ** 2).mean(0) / d).tolist()
    return out
