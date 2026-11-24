"""In-context linear regression with unequal input variances.

A prompt holds n examples (x_i, y_i = w^T x_i) with a fresh task w ~ N(0, I) and inputs x ~ N(0, Lambda),
Lambda = diag(lambda_1, ..., lambda_d) with lambda_j proportional to 1/j^2 and trace d.
"""
import math

import numpy as np
import torch


def spectrum(d):
    lam = np.arange(1, d + 1, dtype=np.float64) ** -2.0
    return lam * d / lam.sum()


def shifted(lam, t):
    """Query spectrum lambda^(1 - 2t) at the same trace: t < 0 toward common directions, t > 0 toward rare ones."""
    q = lam ** (1 - 2 * t)
    return q * lam.sum() / q.sum()


def scale(lam):
    """x = z @ scale(lam) turns z ~ N(0, I) into x ~ N(0, diag(lam))."""
    return torch.tensor(np.diag(np.sqrt(lam)), dtype=torch.float32)


def tasks(x, w):
    return torch.einsum("pnd,pd->pn", x, w)


class Sampler:
    """Training prompts, drawn on the fly."""

    def __init__(self, lam, n, seed):
        self.A, self.n = scale(lam), n
        self.g = torch.Generator().manual_seed(seed)

    def sample(self, batch):
        x = torch.randn(batch, self.n, len(self.A), generator=self.g) @ self.A
        w = torch.randn(batch, len(self.A), 1, generator=self.g)
        return x, (x @ w).squeeze(-1)


def eval_prompts(lam, n, seed, shifts, n_test=1000, n_probe=64, n_shift=256):
    """Fixed evaluation prompts.

    test:  in-distribution prompts.
    probe: n_probe prompts per direction j, with the task along that direction only (w = sqrt(d) e_j).
    shift: one set of tasks and contexts; only the final query changes, drawn from N(0, shifted(lam, t)).
    """
    d = len(lam)
    g = torch.Generator().manual_seed(seed)
    A = scale(lam)

    x = torch.randn(n_test, n, d, generator=g) @ A
    test = dict(x=x, y=tasks(x, torch.randn(n_test, d, generator=g)))

    x = torch.randn(d * n_probe, n, d, generator=g) @ A
    j = torch.arange(d).repeat_interleave(n_probe)
    probe = dict(x=x, y=tasks(x, math.sqrt(d) * torch.eye(d)[j]), j=j)

    context = torch.randn(n_shift, n - 1, d, generator=g) @ A
    zq = torch.randn(n_shift, 1, d, generator=g)
    w = torch.randn(n_shift, d, generator=g)
    xs = torch.stack([torch.cat([context, zq @ scale(shifted(lam, t))], 1) for t in shifts])
    shift = dict(x=xs, y=torch.stack([tasks(x, w) for x in xs]), t=list(shifts))
    return dict(test=test, probe=probe, shift=shift)
