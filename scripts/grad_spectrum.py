"""(python scripts/grad_spectrum.py)  Singular-value spectra of the hidden-matrix gradients on the plateau (init, iso, mean of 16 batches,
and a single minibatch).  If a few directions carry most of the gradient, normalised SGD spends its
step on them, while Muon (orthogonalised) gives every direction the same step."""
import sys
import torch
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from icl.data import make_condition, Sampler
from icl.model import GPT2ICL
from icl.optim import newton_schulz5

torch.manual_seed(0); torch.set_num_threads(1)
cond = make_condition("iso")
model = GPT2ICL(8, 33, y_mode="sep", readin_init="gauss")
smp = Sampler(cond, 33, seed=777)
G1, G = {}, {}
nb = 16
for b in range(nb):
    x, y = smp.sample(64)
    model.zero_grad(); ((model(x, y) - y) ** 2).mean().backward()
    for n, p in model.named_parameters():
        if n.startswith("blocks.") and p.ndim == 2:
            G.setdefault(n, torch.zeros_like(p)).add_(p.grad / nb)
            if b == 0:
                G1[n] = p.grad.clone()
print(f"{'matrix':18s} {'top-1 share':>11s} {'top-8 share':>11s} {'eff. rank':>9s} | {'1 batch: top-1':>14s} {'eff. rank':>9s}")
def stats(M):
    s = torch.linalg.svdvals(M.double())
    e = s ** 2 / (s ** 2).sum()
    return e[0].item(), e[:8].sum().item(), (s.sum() ** 2 / (s ** 2).sum()).item()  # participation-style rank
for n in [k for k in G if k.startswith("blocks.0.") or k.startswith("blocks.1.")]:
    a = stats(G[n]); b = stats(G1[n])
    print(f"{n:18s} {a[0]:11.3f} {a[1]:11.3f} {a[2]:9.1f} | {b[0]:14.3f} {b[2]:9.1f}")

# Is the dominant direction just "rescale the output"? Compare the top left-singular vector of each
# residual-writing matrix (o, proj: rows = residual coordinates) with the read-out weight vector
# (after the final LayerNorm gain), and the top right-singular vector of q/k/v/fc (columns = residual
# input coordinates) with the same vector.
wro = (model.read_out.weight[0] * model.ln_f.weight).detach().double()
wro = wro / wro.norm()
print("\n|cos(top singular vector, read-out direction)|  (random baseline ~ %.2f)" % (1 / 128 ** 0.5))
for n in [k for k in G if k.startswith("blocks.0.") or k.startswith("blocks.3.")]:
    U, S, Vh = torch.linalg.svd(G[n].double(), full_matrices=False)
    vec = U[:, 0] if n.endswith(("o.weight", "proj.weight")) else Vh[0]
    side = "output side" if n.endswith(("o.weight", "proj.weight")) else "input side "
    print(f"  {n:20s} {side}: {abs(float(vec @ wro)):.2f}")
