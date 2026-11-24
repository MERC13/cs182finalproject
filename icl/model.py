"""GPT-2 for in-context regression (the architecture of Garg et al., 2022).

Each example becomes two tokens, [x_i, 0] and [x_i, y_i]; the model predicts y_i at [x_i, 0] from the earlier
examples. Pre-LayerNorm blocks, learned positions, GELU MLP, no dropout. Q, K and V are separate matrices so that
Muon updates each one.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class Block(nn.Module):
    def __init__(self, width, heads, layers):
        super().__init__()
        self.heads = heads
        self.ln_1 = nn.LayerNorm(width)
        self.q, self.k, self.v, self.o = (nn.Linear(width, width) for _ in range(4))
        self.ln_2 = nn.LayerNorm(width)
        self.fc = nn.Linear(width, 4 * width)
        self.proj = nn.Linear(4 * width, width)
        for m in (self.q, self.k, self.v, self.fc):  # GPT-2 initialization
            nn.init.normal_(m.weight, std=0.02)
            nn.init.zeros_(m.bias)
        for m in (self.o, self.proj):
            nn.init.normal_(m.weight, std=0.02 / math.sqrt(2 * layers))
            nn.init.zeros_(m.bias)

    def matrices(self):
        return self.q, self.k, self.v, self.o, self.fc, self.proj

    def forward(self, h):
        B, T, C = h.shape
        a = self.ln_1(h)
        q, k, v = (m(a).float().view(B, T, self.heads, C // self.heads).transpose(1, 2) for m in (self.q, self.k, self.v))
        with torch.autocast("cpu", enabled=False):  # attention in float32
            att = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        h = h + self.o(att.transpose(1, 2).reshape(B, T, C))
        return h + self.proj(F.gelu(self.fc(self.ln_2(h)), approximate="tanh"))


class GPT2(nn.Module):
    def __init__(self, d=20, n=41, width=256, layers=12, heads=8, init_std=0.031):
        super().__init__()
        self.read_in = nn.Linear(d + 1, width)
        nn.init.normal_(self.read_in.weight, std=1 / math.sqrt(3 * (d + 1)))
        self.wpe = nn.Embedding(2 * n, width)
        nn.init.normal_(self.wpe.weight, std=0.02)
        self.blocks = nn.ModuleList(Block(width, heads, layers) for _ in range(layers))
        for block in self.blocks:  # one std for every hidden matrix: with GPT-2's, SGD did not train
            for m in block.matrices():
                nn.init.normal_(m.weight, std=init_std)
        self.ln_f = nn.LayerNorm(width)
        self.read_out = nn.Linear(width, 1)

    def forward(self, x, y):
        """Predictions (B, n): entry k predicts y_k from the k examples before it."""
        query = torch.cat([x, torch.zeros_like(y).unsqueeze(-1)], -1)
        example = torch.cat([x, y.unsqueeze(-1)], -1)
        z = torch.stack([query, example], 2).flatten(1, 2)[:, :-1]
        h = self.read_in(z) + self.wpe.weight[: z.shape[1]]
        for block in self.blocks:
            h = block(h)
        return self.read_out(self.ln_f(h))[:, ::2, 0]

    @classmethod
    def load(cls, path):
        model = cls()
        model.load_state_dict(torch.load(path, map_location="cpu")["model"])
        return model.eval()
