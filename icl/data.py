"""In-context linear regression data, following Garg et al. (2022).

Each prompt holds n_points pairs (x_i, y_i = w^T x_i).  The model predicts
y_i from (x_1, y_1, ..., x_{i-1}, y_{i-1}, x_i), so a single prompt trains
every prefix length k = 0 .. n_points-1 ("N = 32 pairs and a query" ->
n_points = 33).

Inputs follow N(0, U diag(lam) U^T) with lam_i ∝ i^{-alpha}, rescaled so
trace = d (alpha = 0 is the isotropic N(0, I) of Garg et al.).  U is the
identity ("aligned") or a fixed Haar-random rotation ("rotated").
Tasks are fresh w ~ N(0, I_d) for every prompt; noiseless.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import torch


def spectrum(d: int, alpha: float) -> np.ndarray:
    lam = np.arange(1, d + 1, dtype=np.float64) ** (-alpha)
    return lam * d / lam.sum()


def haar_rotation(d: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    q, r = np.linalg.qr(rng.standard_normal((d, d)))
    q = q * np.sign(np.diag(r))
    if np.linalg.det(q) < 0:  # keep it a proper rotation
        q[:, 0] = -q[:, 0]
    return q


@dataclass
class Condition:
    """One training input distribution."""

    d: int = 8
    alpha: float = 0.0
    rotated: bool = False
    rot_seed: int = 1234
    lam: np.ndarray = field(init=False)
    U: np.ndarray = field(init=False)  # columns = principal axes, sorted by lam desc

    def __post_init__(self):
        self.lam = spectrum(self.d, self.alpha)
        self.U = haar_rotation(self.d, self.rot_seed) if self.rotated else np.eye(self.d)

    @property
    def name(self) -> str:
        if self.alpha == 0:
            return "iso"
        a = {1.0: "1/i", 2.0: "1/i2"}.get(float(self.alpha), f"a{self.alpha:g}")
        return f"{a}-{'rot' if self.rotated else 'ali'}"

    @property
    def tag(self) -> str:
        return self.name.replace("/", "").replace("i2", "isq")

    @property
    def Sigma(self) -> np.ndarray:
        return self.U @ np.diag(self.lam) @ self.U.T

    def A(self, lam: np.ndarray | None = None) -> torch.Tensor:
        """x = A z with z ~ N(0, I); A = U diag(sqrt(lam))."""
        lam = self.lam if lam is None else lam
        return torch.tensor(self.U @ np.diag(np.sqrt(lam)), dtype=torch.float32)


CONDITIONS = {
    "iso": dict(alpha=0.0, rotated=False),
    "1/i-ali": dict(alpha=1.0, rotated=False),
    "1/i-rot": dict(alpha=1.0, rotated=True),
    "1/i2-ali": dict(alpha=2.0, rotated=False),
    "1/i2-rot": dict(alpha=2.0, rotated=True),
}


def make_condition(name: str, d: int = 8) -> Condition:
    return Condition(d=d, **CONDITIONS[name])


class Sampler:
    """On-the-fly training prompts (fresh w every prompt)."""

    def __init__(self, cond: Condition, n_points: int, seed: int, pool: torch.Tensor | None = None,
                 noise: float = 0.0, pool_p: torch.Tensor | None = None):
        self.cond, self.n_points = cond, n_points
        self.At = cond.A().T.contiguous()
        self.g = torch.Generator().manual_seed(seed)
        self.pool, self.noise = pool, noise  # pool: (M, d) finite task set (None = fresh w per prompt)
        self.pool_p = pool_p  # task frequencies over the pool (None = uniform)

    def sample(self, batch: int, n_points: int | None = None, d_active: int | None = None):
        """n_points / d_active implement Garg et al.'s curriculum: shorter prompts, and inputs whose
        trailing coordinates are zeroed (their n_dims_truncated; only meaningful for isotropic x)."""
        d = self.cond.d
        n = self.n_points if n_points is None else n_points
        x = torch.randn(batch, n, d, generator=self.g) @ self.At
        if d_active is not None and d_active < d:
            x[..., d_active:] = 0.0
        if self.pool is None:
            w = torch.randn(batch, d, 1, generator=self.g)
        else:
            if self.pool_p is None:
                idx = torch.randint(len(self.pool), (batch,), generator=self.g)
            else:
                idx = torch.multinomial(self.pool_p, batch, replacement=True, generator=self.g)
            w = self.pool[idx].unsqueeze(-1)
        y = (x @ w).squeeze(-1)
        if self.noise:
            y = y + self.noise * torch.randn(y.shape, generator=self.g)
        return x, y


def make_pool(n_tasks: int, d: int, seed: int) -> torch.Tensor:
    """A fixed finite set of training tasks w_1..w_M ~ N(0, I_d) (Raventos et al. 2023)."""
    return torch.randn(n_tasks, d, generator=torch.Generator().manual_seed(seed))


def zipf_probs(n_tasks: int, alpha: float) -> torch.Tensor:
    """Task frequencies p_k ∝ (k + 1)^-alpha: task 0 is the most frequent (Chan et al. 2022)."""
    p = torch.arange(1, n_tasks + 1, dtype=torch.float64) ** -alpha
    return (p / p.sum()).float()


def query_shift_lams(lam: np.ndarray, ts) -> list[np.ndarray]:
    """Query spectra lam^(1-2t), renormalised to the same trace.

    t = 0: training spectrum; t = 0.5: isotropic; t = 1: inverted spectrum,
    i.e. most query variance on the axes that had least variance in training.
    """
    out = []
    for t in ts:
        l = lam ** (1.0 - 2.0 * t)
        out.append(l * lam.sum() / l.sum())
    return out


@dataclass
class EvalSets:
    x: torch.Tensor  # (P, n, d)   random-task prompts
    y: torch.Tensor
    w: torch.Tensor
    px: torch.Tensor  # (d * Pp, n, d) probe prompts, w = sqrt(d) u_k
    py: torch.Tensor
    pdir: torch.Tensor  # (d * Pp,) principal-axis index k of each probe prompt
    qx: torch.Tensor  # (T, Pq, n, d) query-shift prompts (context shared across t)
    qy: torch.Tensor  # (T, Pq, n)
    qts: list
    ix: torch.Tensor  # (Pi * Q, n, d) implied-weight prompts: Pi contexts x Q queries each
    iy: torch.Tensor
    iw: torch.Tensor  # (Pi, d) true task of each context
    iq: torch.Tensor  # (Pi, Q, d) the queries


def make_eval_sets(cond: Condition, n_points: int, seed: int, n_test: int = 1000,
                   n_probe: int = 64, n_qshift: int = 256,
                   qts=(0.0, 0.25, 0.5, 0.75, 1.0), n_iw: int = 64, n_iw_q: int = 16) -> EvalSets:
    d = cond.d
    g = torch.Generator().manual_seed(seed)
    A = cond.A()
    # random-task prompts
    x = torch.randn(n_test, n_points, d, generator=g) @ A.T
    w = torch.randn(n_test, d, generator=g)
    y = torch.einsum("pnd,pd->pn", x, w)
    # probe prompts: task along one principal axis, |w|^2 = d (typical norm)
    U = torch.tensor(cond.U, dtype=torch.float32)
    px = torch.randn(d * n_probe, n_points, d, generator=g) @ A.T
    pdir = torch.arange(d).repeat_interleave(n_probe)
    pw = math.sqrt(d) * U[:, pdir].T  # (d*n_probe, d)
    py = torch.einsum("pnd,pd->pn", px, pw)
    # query-shift prompts: same tasks and contexts, only the final query changes
    zc = torch.randn(n_qshift, n_points - 1, d, generator=g)
    zq = torch.randn(n_qshift, 1, d, generator=g)
    qw = torch.randn(n_qshift, d, generator=g)
    xc = zc @ A.T
    qx, qy = [], []
    for lq in query_shift_lams(cond.lam, qts):
        xq = zq @ cond.A(lq).T
        xx = torch.cat([xc, xq], 1)
        qx.append(xx)
        qy.append(torch.einsum("pnd,pd->pn", xx, qw))
    # implied-weight prompts: one context, many in-distribution queries
    ic = torch.randn(n_iw, n_points - 1, d, generator=g) @ A.T
    iw = torch.randn(n_iw, d, generator=g)
    iq = torch.randn(n_iw, n_iw_q, d, generator=g) @ A.T
    ix = torch.cat([ic.unsqueeze(1).expand(-1, n_iw_q, -1, -1), iq.unsqueeze(2)], 2).reshape(n_iw * n_iw_q, n_points, d)
    iy = torch.einsum("pnd,pd->pn", ix, iw.repeat_interleave(n_iw_q, 0))
    return EvalSets(x, y, w, px, py, pdir, torch.stack(qx), torch.stack(qy), list(qts), ix, iy, iw, iq)
