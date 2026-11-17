"""Minimal GPT-2 matching Garg et al.'s TransformerModel.

Garg et al. wrap HuggingFace GPT2Model (pre-LayerNorm blocks, learned
positional embeddings, gelu_new MLP with 4x width, dropout 0) between a
Linear read-in and a Linear(n_embd, 1) read-out that keep PyTorch's default
init, while GPT-2 weights use N(0, 0.02) and residual projections
N(0, 0.02 / sqrt(2 n_layer)).

Differences from HF, both function-preserving:
  * Q, K and V are separate matrices instead of one fused c_attn, so Muon can
    orthogonalise each of them (the modded-nanogpt convention).  The init
    distribution is identical.
  * The last y token is dropped: with causal attention it never affects a
    prediction at an x position.

y_mode = "garg": y is written into coordinate 0 of a d-dim token that shares
the read-in with x (Garg's _combine).  y_mode = "sep": tokens are d+1 dims,
x -> [x, 0] and y -> [0, y], so y has its own input coordinate.  y_mode = "concat": the y token
carries its x as well, [x, y], as in the linear-attention ICL theory (von Oswald et al., Ahn et al.),
so one attention step can read (x_i, y_i) pairs without first copying x_i into the y token.
readin_init = "gauss" draws the read-in from N(0, 1/(3 fan_in)) (same std as
the default Kaiming-uniform).  With y_mode = "sep" this makes SGD -- and Muon
if it also updates the read-in -- exactly equivariant to rotating x, so the
aligned and rotated conditions are equal in distribution for those optimizers.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class Block(nn.Module):
    def __init__(self, n_embd: int, n_head: int, n_layer: int):
        super().__init__()
        self.n_head = n_head
        self.ln_1 = nn.LayerNorm(n_embd, eps=1e-5)
        self.q = nn.Linear(n_embd, n_embd)
        self.k = nn.Linear(n_embd, n_embd)
        self.v = nn.Linear(n_embd, n_embd)
        self.o = nn.Linear(n_embd, n_embd)
        self.ln_2 = nn.LayerNorm(n_embd, eps=1e-5)
        self.fc = nn.Linear(n_embd, 4 * n_embd)
        self.proj = nn.Linear(4 * n_embd, n_embd)
        for m in (self.q, self.k, self.v, self.fc):
            nn.init.normal_(m.weight, std=0.02)
            nn.init.zeros_(m.bias)
        for m in (self.o, self.proj):  # GPT-2 scaled residual init
            nn.init.normal_(m.weight, std=0.02 / math.sqrt(2 * n_layer))
            nn.init.zeros_(m.bias)

    def forward(self, h):
        B, T, C = h.shape
        a = self.ln_1(h)
        q, k, v = (m(a).float().view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
                   for m in (self.q, self.k, self.v))
        # attention in fp32 even under bf16 autocast (the CPU bf16 kernel is ~5x slower)
        with torch.autocast("cpu", enabled=False):
            att = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        h = h + self.o(att.transpose(1, 2).reshape(B, T, C))
        return h + self.proj(F.gelu(self.fc(self.ln_2(h)), approximate="tanh"))


class GPT2ICL(nn.Module):
    def __init__(self, d: int, n_points: int, n_embd: int = 128, n_layer: int = 4,
                 n_head: int = 4, y_mode: str = "garg", readin_init: str = "torch", hidden_init: float = 0.0):
        super().__init__()
        assert y_mode in ("garg", "sep", "concat") and readin_init in ("torch", "gauss", "eye")
        self.d, self.y_mode = d, y_mode
        fan_in = d if y_mode == "garg" else d + 1
        self.read_in = nn.Linear(fan_in, n_embd)  # torch default (Kaiming-uniform) init, as in Garg
        if readin_init == "gauss":
            # same per-entry std as Kaiming-uniform, but rotation-invariant in distribution
            nn.init.normal_(self.read_in.weight, std=1.0 / math.sqrt(3 * fan_in))
        elif readin_init == "eye":
            # input coordinate j -> residual coordinate j (zeros elsewhere): the hidden layers see the
            # data's own basis, so an axis-aligned covariance stays axis-aligned for Adam downstream
            with torch.no_grad():
                self.read_in.weight.zero_()
                self.read_in.weight[:fan_in, :fan_in] = torch.eye(fan_in)
        self.wpe = nn.Embedding(2 * n_points, n_embd)
        nn.init.normal_(self.wpe.weight, std=0.02)
        self.blocks = nn.ModuleList(Block(n_embd, n_head, n_layer) for _ in range(n_layer))
        if hidden_init:  # one std for every hidden matrix instead of GPT-2's 0.02 and 0.02/sqrt(2L) (SGD-rescue test)
            for blk in self.blocks:
                for m in (blk.q, blk.k, blk.v, blk.o, blk.fc, blk.proj):
                    nn.init.normal_(m.weight, std=hidden_init)
        self.ln_f = nn.LayerNorm(n_embd, eps=1e-5)
        self.read_out = nn.Linear(n_embd, 1)  # torch default init

    def tokens(self, x, y):
        B, n, d = x.shape
        if self.y_mode == "garg":
            xt = x
            yt = torch.cat([y.unsqueeze(-1), x.new_zeros(B, n, d - 1)], -1)
        elif self.y_mode == "sep":
            xt = torch.cat([x, x.new_zeros(B, n, 1)], -1)
            yt = torch.cat([x.new_zeros(B, n, d), y.unsqueeze(-1)], -1)
        else:  # concat: query token [x_k, 0], context token [x_k, y_k] (no copy step needed)
            xt = torch.cat([x, x.new_zeros(B, n, 1)], -1)
            yt = torch.cat([x, y.unsqueeze(-1)], -1)
        z = torch.stack([xt, yt], 2).reshape(B, 2 * n, -1)
        return z[:, :-1]  # drop the final y token

    def forward(self, x, y):
        """Returns predictions (B, n): entry k predicts y_{k+1} from k examples."""
        z = self.tokens(x, y)
        h = self.read_in(z) + self.wpe.weight[: z.shape[1]]
        for blk in self.blocks:
            h = blk(h)
        return self.read_out(self.ln_f(h))[:, ::2, 0]
