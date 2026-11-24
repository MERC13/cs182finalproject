"""Muon (Jordan et al., 2024) and the spectral knob.

Muon takes Nesterov momentum and replaces the update M = U S V^T by approximately U V^T, computed with five
Newton-Schulz iterations. With `power` set, an exact SVD replaces Newton-Schulz and keeps S^power, rescaled to the
size of an orthogonal update: power 0 is exact orthogonalization, power 1 the normalized momentum.
"""
import torch


def newton_schulz(G, steps=5, eps=1e-7):
    a, b, c = 3.4445, -4.7750, 2.0315
    X = G.float()
    tall = X.size(-2) > X.size(-1)
    if tall:
        X = X.mT
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + eps)
    for _ in range(steps):
        A = X @ X.mT
        X = a * X + (b * A + c * (A @ A)) @ X
    return X.mT if tall else X


def polar(G):
    U, _, Vh = torch.linalg.svd(G.double(), full_matrices=False)
    return (U @ Vh).to(G.dtype)


def spectral_power(G, power):
    U, S, Vh = torch.linalg.svd(G.double(), full_matrices=False)
    X = (U * S.clamp_min(0) ** power) @ Vh
    X = X * (min(G.shape) ** 0.5 / (X.norm() + 1e-30))
    return X.to(G.dtype)


class Muon(torch.optim.Optimizer):
    def __init__(self, params, lr, momentum=0.95, power=None):
        super().__init__(params, dict(lr=lr, momentum=momentum, power=power))

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            beta, power = group["momentum"], group["power"]
            for p in group["params"]:
                state = self.state[p]
                if "buf" not in state:
                    state["buf"] = torch.zeros_like(p.grad)
                buf = state["buf"].lerp_(p.grad, 1 - beta)
                u = p.grad.lerp(buf, beta)  # Nesterov
                if power is None:
                    u = newton_schulz(u)
                elif power == 0:
                    u = polar(u)
                else:
                    u = spectral_power(u, power)
                u = u * max(1, p.size(-2) / p.size(-1)) ** 0.5
                p.add_(u.to(p.dtype), alpha=-group["lr"])


class Chain:
    """Several optimizers stepped as one."""

    def __init__(self, *opts):
        self.opts = opts
        self.param_groups = [g for o in opts for g in o.param_groups]

    def zero_grad(self):
        for o in self.opts:
            o.zero_grad()

    def step(self):
        for o in self.opts:
            o.step()


def make_optimizer(model, name, lr, power=None):
    if name == "sgd":
        return torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    if name == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.0)
    if name == "muon":  # hidden weight matrices; AdamW for embeddings, read-in/out, gains and biases
        hidden = [p for n, p in model.named_parameters() if n.startswith("blocks.") and p.ndim == 2]
        rest = [p for n, p in model.named_parameters() if not (n.startswith("blocks.") and p.ndim == 2)]
        return Chain(Muon(hidden, lr, power=power), torch.optim.AdamW(rest, lr=1e-3, weight_decay=0.0))
    raise ValueError(name)
