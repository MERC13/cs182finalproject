"""SGD+momentum, AdamW and Muon (+ auxiliary AdamW) for the ICL pilot.

Muon follows Keller Jordan's reference implementation (github.com/KellerJordan/Muon):
momentum buffer m <- beta m + (1-beta) g, Nesterov update g + beta (m - g)...
expressed with lerp exactly as in the reference, then 5 quintic Newton-Schulz
iterations with coefficients (3.4445, -4.7750, 2.0315) after Frobenius
normalisation, scaled by sqrt(max(1, rows/cols)).  We run Newton-Schulz in
float32 (the reference uses bfloat16 on GPU).  "exact" replaces Newton-Schulz
by the SVD polar factor U V^T (the "simplified Muon" of the theory papers).
Ablations: "fro" rescales the momentum to the same per-matrix Frobenius norm instead of
orthogonalising it (per-matrix normalised SGD); "none" applies it unnormalised (SGD on the
hidden matrices, AdamW on the rest).

Parameter routing for Muon (`muon_params`):
  "hidden"  : 2-D weights inside transformer blocks (q, k, v, o, fc, proj)
              -> Muon; read-in, read-out, positional embedding, biases and
              LayerNorm gains -> AdamW.  This is the reference recipe.
  "hidden+in": additionally the read-in Linear(d, n_embd).
"""
from __future__ import annotations

import math

import torch


@torch.no_grad()
def newton_schulz5(G: torch.Tensor, steps: int = 5, eps: float = 1e-7) -> torch.Tensor:
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.float()
    tall = X.size(-2) > X.size(-1)
    if tall:
        X = X.mT
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + eps)
    for _ in range(steps):
        A = X @ X.mT
        B = b * A + c * (A @ A)
        X = a * X + B @ X
    return X.mT if tall else X


@torch.no_grad()
def polar_exact(G: torch.Tensor) -> torch.Tensor:
    U, _, Vh = torch.linalg.svd(G.double(), full_matrices=False)
    return (U @ Vh).to(G.dtype)


@torch.no_grad()
def spectral_power(G: torch.Tensor, power: float) -> torch.Tensor:
    """U diag(s^power) V^T, rescaled to the Frobenius norm of an orthogonal update (sqrt(min(m, n))).

    power = 1 keeps the gradient's own singular values (per-matrix normalised SGD, the "fro" ablation);
    power = 0 is the exact polar factor (Muon with exact orthogonalisation). In between, it is steepest
    descent in the Schatten-q norm with q = 1 + 1/power (Frobenius at power 1, spectral at power 0)."""
    U, S, Vh = torch.linalg.svd(G.double(), full_matrices=False)
    X = (U * S.clamp_min(0) ** power) @ Vh
    X = X * (min(G.size(-2), G.size(-1)) ** 0.5 / (X.norm() + 1e-30))
    return X.to(G.dtype)


class Muon(torch.optim.Optimizer):
    def __init__(self, params, lr=0.02, momentum=0.95, nesterov=True, ns_steps=5,
                 weight_decay=0.0, orth="ns", power=0.0):
        super().__init__(params, dict(lr=lr, momentum=momentum, nesterov=nesterov,
                                      ns_steps=ns_steps, weight_decay=weight_decay, orth=orth, power=power))

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            beta = group["momentum"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad
                st = self.state[p]
                if "momentum_buffer" not in st:
                    st["momentum_buffer"] = torch.zeros_like(g)
                buf = st["momentum_buffer"]
                buf.lerp_(g, 1 - beta)
                u = g.lerp(buf, beta) if group["nesterov"] else buf
                if group["orth"] == "ns":
                    u = newton_schulz5(u, group["ns_steps"])
                elif group["orth"] == "exact":
                    u = polar_exact(u)
                elif group["orth"] == "pow":
                    # spectral-flattening knob: singular values raised to `power` (1 = gradient, 0 = polar)
                    u = spectral_power(u, group["power"])
                elif group["orth"] == "fro":
                    # ablation: per-matrix normalised SGD. Same Frobenius step size as an orthogonal
                    # update (min(rows, cols) unit singular values), but the gradient's own direction,
                    # so its singular-value spectrum (GD's sequential bias) is kept.
                    u = u * (min(p.size(-2), p.size(-1)) ** 0.5 / (u.norm() + 1e-12))
                elif group["orth"] != "none":  # "none": plain momentum SGD on these matrices
                    raise ValueError(group["orth"])
                if group["orth"] != "none":
                    u = u * max(1.0, p.size(-2) / p.size(-1)) ** 0.5
                if group["weight_decay"]:
                    p.mul_(1 - group["lr"] * group["weight_decay"])
                p.add_(u.to(p.dtype), alpha=-group["lr"])


class Combined:
    """Steps several optimizers as one; exposes all param groups for LR scheduling."""

    def __init__(self, opts):
        self.opts = opts

    def state_dict(self):
        return [o.state_dict() for o in self.opts]

    def load_state_dict(self, states):
        for o, st in zip(self.opts, states):
            o.load_state_dict(st)

    @property
    def param_groups(self):
        return [g for o in self.opts for g in o.param_groups]

    def zero_grad(self, set_to_none=True):
        for o in self.opts:
            o.zero_grad(set_to_none=set_to_none)

    def step(self):
        for o in self.opts:
            o.step()


def _haar(n, g):
    q, r = torch.linalg.qr(torch.randn(n, n, generator=g, dtype=torch.float64))
    return (q * torch.sign(torch.diagonal(r))).float()


class RotatedAdam:
    """Adam in a randomly rotated parameter space, as in Xie et al. (2025).

    theta = R phi on the flattened parameter vector (zero-padded to a^3 entries), with
    R = P2 (Q1 (x) Q2 (x) Q3) P1: two random permutations around a Kronecker product of three
    Haar-orthogonal a x a matrices.  phi_0 = R^T theta_0, so the network function and the loss
    landscape are unchanged; only Adam's coordinate system is rotated.  The padded coordinates
    are inert (zero gradient in theta-space).
    """

    def __init__(self, params, lr, betas=(0.9, 0.999), eps=1e-8, seed=0):
        self.params = [p for p in params if p.requires_grad]
        self.sizes = [p.numel() for p in self.params]
        self.P = sum(self.sizes)
        self.a = int(math.ceil(self.P ** (1 / 3) - 1e-9))
        while self.a ** 3 < self.P:
            self.a += 1
        self.N = self.a ** 3
        g = torch.Generator().manual_seed(10_000 + seed)
        self.p1, self.p2 = torch.randperm(self.N, generator=g), torch.randperm(self.N, generator=g)
        self.i1, self.i2 = torch.argsort(self.p1), torch.argsort(self.p2)
        self.Q = [_haar(self.a, g) for _ in range(3)]
        self.betas, self.eps, self.t = betas, eps, 0
        self.param_groups = [dict(lr=lr, base_lr=lr)]
        with torch.no_grad():
            self.phi = self.Rt(self._flat([p.detach() for p in self.params]))
        self.m, self.v = torch.zeros(self.N), torch.zeros(self.N)

    def _kron(self, z, Qs):
        a = self.a
        Z = z.view(a, a, a)
        Z = torch.einsum("ia,abc->ibc", Qs[0], Z)
        Z = torch.einsum("jb,abc->ajc", Qs[1], Z)
        Z = torch.einsum("kc,abc->abk", Qs[2], Z)
        return Z.reshape(-1)

    def R(self, z):
        return self._kron(z[self.p1], self.Q)[self.p2]

    def Rt(self, z):
        return self._kron(z[self.i2], [q.T for q in self.Q])[self.i1]

    def _flat(self, tensors):
        out = torch.zeros(self.N)
        out[: self.P] = torch.cat([t.reshape(-1).float().cpu() for t in tensors])
        return out

    def zero_grad(self, set_to_none=True):
        for p in self.params:
            p.grad = None

    @torch.no_grad()
    def step(self):
        g = self.Rt(self._flat([p.grad if p.grad is not None else torch.zeros_like(p) for p in self.params]))
        b1, b2 = self.betas
        self.t += 1
        self.m.mul_(b1).add_(g, alpha=1 - b1)
        self.v.mul_(b2).addcmul_(g, g, value=1 - b2)
        mh, vh = self.m / (1 - b1 ** self.t), self.v / (1 - b2 ** self.t)
        self.phi -= self.param_groups[0]["lr"] * mh / (vh.sqrt() + self.eps)
        theta, off = self.R(self.phi), 0
        for p, n in zip(self.params, self.sizes):
            p.copy_(theta[off:off + n].view_as(p).to(p.device, p.dtype))
            off += n

    def state_dict(self):
        return dict(phi=self.phi, m=self.m, v=self.v, t=self.t)

    def load_state_dict(self, st):
        self.phi, self.m, self.v, self.t = st["phi"], st["m"], st["v"], st["t"]


def split_params(model, muon_params: str = "hidden"):
    muon, other = [], []
    for name, p in model.named_parameters():
        in_block = name.startswith("blocks.") and p.ndim == 2
        is_readin = name == "read_in.weight"
        if in_block or (muon_params == "hidden+in" and is_readin):
            muon.append(p)
        else:
            other.append(p)
    return muon, other


def build_optimizer(model, opt: str, lr: float, aux_lr: float = 1e-3, momentum: float = 0.9,
                    weight_decay: float = 0.0, betas=(0.9, 0.999), muon_params: str = "hidden",
                    muon_orth: str = "ns", muon_momentum: float = 0.95, nesterov: bool = False,
                    param_rotation: bool = False, seed: int = 0, muon_power: float = 0.0):
    if param_rotation:
        assert opt == "adamw" and weight_decay == 0, "rotated-parameter run is defined for Adam(W) without decay"
        o = RotatedAdam(model.parameters(), lr=lr, betas=betas, seed=seed)
    elif opt == "sgd":
        o = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum, nesterov=nesterov,
                            weight_decay=weight_decay)
    elif opt == "adamw":
        o = torch.optim.AdamW(model.parameters(), lr=lr, betas=betas, weight_decay=weight_decay)
    elif opt == "muon":
        mp, op = split_params(model, muon_params)
        o = Combined([
            Muon(mp, lr=lr, momentum=muon_momentum, weight_decay=weight_decay, orth=muon_orth, power=muon_power),
            torch.optim.AdamW(op, lr=aux_lr, betas=betas, weight_decay=weight_decay),
        ])
    else:
        raise ValueError(opt)
    for g in o.param_groups:
        g["base_lr"] = g["lr"]
    return o
