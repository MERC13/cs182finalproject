"""(python scripts/grad_blocks.py)  Per-block gradient norms of the GPT-2 on the ICL plateau (init), averaged over batches.
Shows why one global SGD learning rate cannot move the attention circuit."""
import re, sys
import torch
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from icl.data import make_condition, Sampler
from icl.model import GPT2ICL

torch.manual_seed(0); torch.set_num_threads(1)
cond = make_condition("iso")
model = GPT2ICL(8, 33, y_mode="sep", readin_init="gauss")
smp = Sampler(cond, 33, seed=777)

def group(name):
    if name.startswith("blocks."):
        m = re.match(r"blocks\.\d+\.(\w+)\.(weight|bias)", name)
        kind, wb = m.group(1), m.group(2)
        if kind.startswith("ln"):
            return "layernorm (all)"
        return f"{kind}.{wb}"
    return name

G, nb = {}, 16
for _ in range(nb):
    x, y = smp.sample(64)
    model.zero_grad()
    loss = ((model(x, y) - y) ** 2).mean()
    loss.backward()
    for n, p in model.named_parameters():
        G.setdefault(n, torch.zeros_like(p)).add_(p.grad / nb)   # mean gradient (signal)
groups = {}
for n, p in model.named_parameters():
    g = group(n)
    groups.setdefault(g, [0.0, 0])
    groups[g][0] += G[n].pow(2).sum().item(); groups[g][1] += p.numel()
tot = sum(v[0] for v in groups.values()) ** 0.5
print(f"loss at init {loss.item()/8:.3f} (normalised); total mean-gradient norm {tot:.3g}")
print(f"{'block':18s} {'params':>7s} {'|grad|':>9s} {'share of |grad|^2':>17s} {'|grad| per param (rms)':>23s}")
for g, (s2, n) in sorted(groups.items(), key=lambda t: -t[1][0]):
    print(f"{g:18s} {n:7d} {s2**0.5:9.2e} {s2/tot**2:17.2e} {(s2/n)**0.5:23.2e}")
qk = (groups["q.weight"][0] + groups["k.weight"][0]) ** 0.5
ro = groups["read_out.weight"][0] ** 0.5
print(f"\nquery+key weight gradient / read-out gradient = {qk/ro:.2e}")

# relative change per step (|update| / |weights|) that each optimizer applies to each block at init
print(f"\n{'block':18s} {'|W|':>7s} {'SGD lr0.1 (mom 0.9)':>20s} {'AdamW lr5e-4':>13s} {'Muon lr0.005':>13s}")
P = dict(model.named_parameters())
for kind in ["read_out.weight", "read_in.weight", "q.weight", "k.weight", "v.weight", "o.weight", "fc.weight", "proj.weight"]:
    names = [n for n in P if n == kind or n.endswith("." + kind)]
    W2 = sum(P[n].pow(2).sum().item() for n in names); g2 = sum(G[n].pow(2).sum().item() for n in names)
    numel = sum(P[n].numel() for n in names)
    sgd = 0.1 * 10 * g2 ** 0.5                     # heavy-ball steady state: lr * |g| / (1 - 0.9)
    adam = 5e-4 * numel ** 0.5                      # ~lr per coordinate
    if kind.startswith(("read_", )):
        muon = 1e-3 * numel ** 0.5                  # aux AdamW (lr 1e-3) on read-in/out
    else:
        muon = sum(0.005 * min(P[n].shape) ** 0.5 * max(1, P[n].shape[0] / P[n].shape[1]) ** 0.5 for n in names) / len(names) * len(names) ** 0.5
    w = W2 ** 0.5
    print(f"{kind:18s} {w:7.3f} {sgd / w:20.2e} {adam / w:13.2e} {muon / w:13.2e}")
