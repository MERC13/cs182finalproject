"""Reference in-context estimators, evaluated at every prefix length.

All functions take context X (P, k, d), Y (P, k), query xq (P, d) and return
predictions (P,).  `prefix_preds` evaluates one at every prefix k = 0..n-1
(k = 0 -> predict 0, the prior mean).

  ols        min-norm least squares (pinv). Noiseless + w ~ N(0, I): this is
             the Bayes posterior mean for every k, and exact once k >= d.
  ridge(l)   (X^T X + l I)^{-1} X^T y.  l -> 0 recovers ols.
  pcr(m)     the best predictor that only sees the top-m POPULATION principal
             axes U_m: ridge in that subspace with penalty s2 = sum_{j>m} lam_j
             (the omitted directions act as noise of that variance).  Plain
             least squares in the subspace explodes near k = m, so we use this
             Bayes version; m = d recovers ols.
  gd_pre(m)  one step of preconditioned GD from w = 0 restricted to the top-m
             axes, w = sum_{i<=m} c_i u_i u_i^T (1/k) X^T y, with the finite-k
             optimal c_i = k / ((k+1) lam_i + tr Sigma)  (from
             E[S A S] = (k+1)/k Sigma A Sigma + tr(A Sigma) Sigma / k for the
             empirical covariance S).  This is what a linear-attention layer
             that has learned m principal axes can compute (Zhang et al. 2025).
  gd1        plain one-step GD with the step size that is optimal on average
             (von Oswald et al. 2023 style), fitted per k.
"""
from __future__ import annotations

import numpy as np
import torch


def ols(X, Y, xq):
    w = torch.linalg.pinv(X) @ Y.unsqueeze(-1)
    return (w.squeeze(-1) * xq).sum(-1)


def ridge(lam):
    def f(X, Y, xq):
        d = X.shape[-1]
        G = X.mT @ X + lam * torch.eye(d, dtype=X.dtype)
        w = torch.linalg.solve(G, X.mT @ Y.unsqueeze(-1))
        return (w.squeeze(-1) * xq).sum(-1)
    return f


def pcr(U_m, s2):
    def f(X, Y, xq):
        Z = X @ U_m
        m = Z.shape[-1]
        if s2 <= 0:
            wz = torch.linalg.pinv(Z) @ Y.unsqueeze(-1)
        else:
            wz = torch.linalg.solve(Z.mT @ Z + s2 * torch.eye(m, dtype=Z.dtype), Z.mT @ Y.unsqueeze(-1))
        return ((xq @ U_m) * wz.squeeze(-1)).sum(-1)
    return f


def gd_pre(U_m, lam_m, trace):
    def f(X, Y, xq):
        k = X.shape[1]
        c = k / ((k + 1) * lam_m + trace)
        P = U_m @ torch.diag(c) @ U_m.mT
        g = (X.mT @ Y.unsqueeze(-1)).squeeze(-1) / k
        return ((g @ P) * xq).sum(-1)
    return f


def prefix_preds(X, Y, f):
    P, n, d = X.shape
    out = torch.zeros(P, n, dtype=torch.float64)
    for k in range(1, n):
        out[:, k] = f(X[:, :k], Y[:, :k], X[:, k])
    return out


RIDGE_GRID = [1e-3, 3e-3, 1e-2, 3e-2, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0]


class Reference:
    """Precomputed estimator predictions on a fixed prompt set."""

    def __init__(self, X, Y, lam: np.ndarray, U: np.ndarray):
        X, Y = X.double(), Y.double()
        d = X.shape[-1]
        U = torch.tensor(U, dtype=torch.float64)
        lam_t = torch.tensor(lam, dtype=torch.float64)
        self.names, preds = [], []

        def add(name, f):
            self.names.append(name)
            preds.append(prefix_preds(X, Y, f))

        add("ols", ols)
        for l in RIDGE_GRID:
            add(f"ridge_{l:g}", ridge(l))
        for m in range(1, d + 1):
            add(f"pcr_{m}", pcr(U[:, :m], float(lam_t[m:].sum())))
        for m in range(1, d + 1):
            add(f"gdpre_{m}", gd_pre(U[:, :m], lam_t[:m], float(lam_t.sum())))
        # one-step plain GD with the per-k optimal step (fit by least squares)
        g1 = []
        for k in range(1, X.shape[1]):
            g = (X[:, :k].mT @ Y[:, :k].unsqueeze(-1)).squeeze(-1) / k
            base = (g * X[:, k]).sum(-1)
            eta = (base * Y[:, k]).sum() / (base * base).sum()
            g1.append(eta * base)
        gd1 = torch.zeros_like(Y)
        gd1[:, 1:] = torch.stack(g1, 1)
        self.names.append("gd1")
        preds.append(gd1)
        self.preds = torch.stack(preds).float()  # (E, P, n)
        self.y = Y.float()
        self.d = d

    def distances(self, model_pred):
        """Normalised squared distance (/d) to each estimator, per prefix: (E, n)."""
        return ((self.preds - model_pred.unsqueeze(0)) ** 2).mean(1) / self.d

    def errors(self):
        """Each estimator's own normalised error vs. the truth, per prefix: (E, n)."""
        return ((self.preds - self.y.unsqueeze(0)) ** 2).mean(1) / self.d
