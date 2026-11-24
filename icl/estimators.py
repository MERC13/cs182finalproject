"""Reference in-context estimators, evaluated on the same prompts as the model.

Each estimator maps a context X (P, k, d), Y (P, k) and query xq (P, d) to predictions (P,).
  ridge(a)    (X^T X + a I)^-1 X^T y
  pcr(m)      ridge on the top m input directions, with penalty sum_{j>m} lambda_j: the omitted directions act as
              noise of that variance (plain least squares on m directions blows up when k is close to m)
  gd_pre(m)   one step of gradient descent preconditioned by the best one-layer linear attention on m directions
              (Zhang et al., 2025)
  gd1         one plain gradient step with the best step size for each k
"""
import torch

RIDGE = [1e-3, 3e-3, 1e-2, 3e-2, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0]


def ols(X, Y, xq):
    w = torch.linalg.pinv(X) @ Y.unsqueeze(-1)
    return (w.squeeze(-1) * xq).sum(-1)


def ridge(a):
    def f(X, Y, xq):
        G = X.mT @ X + a * torch.eye(X.shape[-1], dtype=X.dtype)
        w = torch.linalg.solve(G, X.mT @ Y.unsqueeze(-1))
        return (w.squeeze(-1) * xq).sum(-1)
    return f


def pcr(lam, m):
    if m == len(lam):
        return ols

    def f(X, Y, xq):
        Z = X[..., :m]
        w = torch.linalg.solve(Z.mT @ Z + float(lam[m:].sum()) * torch.eye(m, dtype=Z.dtype), Z.mT @ Y.unsqueeze(-1))
        return (xq[..., :m] * w.squeeze(-1)).sum(-1)
    return f


def gd_pre(lam, m):
    def f(X, Y, xq):
        k = X.shape[1]
        c = torch.zeros_like(lam)
        c[:m] = k / ((k + 1) * lam[:m] + lam.sum())
        g = (X.mT @ Y.unsqueeze(-1)).squeeze(-1) / k
        return (g * c * xq).sum(-1)
    return f


def at_every_k(X, Y, f):
    """Predictions (P, n) for y_k from the first k examples, k = 0 .. n-1 (k = 0: predict 0)."""
    out = torch.zeros(X.shape[:2], dtype=torch.float64)
    for k in range(1, X.shape[1]):
        out[:, k] = f(X[:, :k], Y[:, :k], X[:, k])
    return out


def gd1(X, Y):
    out = torch.zeros(X.shape[:2], dtype=torch.float64)
    for k in range(1, X.shape[1]):
        base = ((X[:, :k].mT @ Y[:, :k].unsqueeze(-1)).squeeze(-1) / k * X[:, k]).sum(-1)
        out[:, k] = (base * Y[:, k]).sum() / (base * base).sum() * base
    return out


def references(X, Y, lam):
    """All reference predictions (name -> (P, n)) on the prompts X, Y."""
    X, Y = X.double(), Y.double()
    lam = torch.tensor(lam, dtype=torch.float64)
    d = X.shape[-1]
    refs = {"ols": at_every_k(X, Y, ols)}
    refs.update({f"ridge_{a:g}": at_every_k(X, Y, ridge(a)) for a in RIDGE})
    refs.update({f"pcr_{m}": at_every_k(X, Y, pcr(lam, m)) for m in range(1, d + 1)})
    refs.update({f"gdpre_{m}": at_every_k(X, Y, gd_pre(lam, m)) for m in range(1, d + 1)})
    refs["gd1"] = gd1(X, Y)
    return refs
