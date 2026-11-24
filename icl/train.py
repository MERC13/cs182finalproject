"""Train the 12-layer GPT-2 and log its evaluations to a JSON file; the final weights go next to it (.pt).

  python -m icl.train --opt muon --lr 0.007 --steps 3000 --seed 10 --out runs/garg_scale/muon_s10.json

--power p replaces Muon's Newton-Schulz step by the spectral knob (p = 0: exact orthogonalization).
"""
import argparse
import json
import math

import numpy as np
import torch

from .data import Sampler, eval_prompts, spectrum
from .estimators import references
from .model import GPT2
from .optim import make_optimizer

D, N = 20, 41  # input dimension; 40 examples and a query
BATCH, WARMUP = 64, 100
EVAL_SEED = 30000
SHIFTS = (0.0, -1.0, -0.5, 0.25, 0.5, 0.75, 1.0)


def eval_steps(steps):
    """Every 100 steps, every 50 up to step 2,000, and steps 25 and 50."""
    return {25, 50} | set(range(0, steps + 1, 100)) | set(range(0, min(2000, steps) + 1, 50))


@torch.no_grad()
def predict(model, x, y, chunk=500):
    dev = next(model.parameters()).device
    return torch.cat([model(x[i:i + chunk].to(dev), y[i:i + chunk].to(dev)).float().cpu() for i in range(0, len(x), chunk)])


def rounded(a):
    a = np.asarray(a, dtype=np.float64)
    return np.vectorize(lambda v: float(f"{v:.5g}"))(a).tolist() if a.ndim else float(f"{a:.5g}")


@torch.no_grad()
def evaluate(model, ev, ref_names, ref_preds, shift):
    """Error after every number of examples; distance to each reference estimator (predictions with 8 or more
    examples); relative probe error per direction; relative error on the shifted queries."""
    model.eval()
    test, probe = ev["test"], ev["probe"]
    pred = predict(model, test["x"], test["y"])
    out = dict(test_err_k=rounded(((pred - test["y"]) ** 2).mean(0) / D))
    dist = (((ref_preds - pred) ** 2).mean(1) / D)[:, 8:].mean(1)
    out["dist_8_32"] = dict(zip(ref_names, rounded(dist)))
    pred = predict(model, probe["x"], probe["y"])
    j, y = probe["j"], probe["y"]
    out["probe_err_k"] = rounded(torch.stack([((pred[j == i] - y[j == i]) ** 2).sum(0) / (y[j == i] ** 2).sum(0)
                                              for i in range(D)]))
    if shift:
        x, y = ev["shift"]["x"], ev["shift"]["y"]
        pred = predict(model, x.flatten(0, 1), y.flatten(0, 1)).view(y.shape)
        out["qshift_err"] = rounded(((pred[..., -1] - y[..., -1]) ** 2).sum(1) / (y[..., -1] ** 2).sum(1))
    model.train()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--opt", required=True, choices=["sgd", "adamw", "muon"])
    ap.add_argument("--lr", type=float, required=True)
    ap.add_argument("--power", type=float, default=None, help="spectral knob for Muon")
    ap.add_argument("--clip", type=float, default=0.0, help="gradient-norm clipping (0: off)")
    ap.add_argument("--steps", type=int, required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cpu")
    a = ap.parse_args()

    torch.set_num_threads(1)
    torch.manual_seed(a.seed)
    lam = spectrum(D)
    dev = torch.device(a.device)
    model = GPT2(D, N).to(dev)
    opt = make_optimizer(model, a.opt, a.lr, a.power)
    base_lr = [g["lr"] for g in opt.param_groups]
    sampler = Sampler(lam, N, a.seed + 777)
    ev = eval_prompts(lam, N, EVAL_SEED, SHIFTS)
    refs = references(ev["test"]["x"], ev["test"]["y"], lam)
    ref_names, ref_preds = list(refs), torch.stack(list(refs.values())).float()
    log = dict(config=vars(a), evals=[], status="running")

    def save():
        with open(a.out, "w") as f:
            json.dump(log, f)

    losses, evals = [], eval_steps(a.steps)
    for step in range(a.steps + 1):
        if step in evals:
            rec = dict(step=step, train_loss=rounded(np.mean(losses) / D) if losses else None)
            rec.update(evaluate(model, ev, ref_names, ref_preds, shift=step % 100 == 0 or step == a.steps))
            log["evals"].append(rec)
            losses = []
            if step % 500 == 0:
                save()
        if step == a.steps:
            log["status"] = "done"
            break
        for g, lr in zip(opt.param_groups, base_lr):
            g["lr"] = lr * min(1.0, (step + 1) / WARMUP)
        x, y = sampler.sample(BATCH)
        x, y = x.to(dev), y.to(dev)
        with torch.autocast(dev.type, dtype=torch.bfloat16, enabled=dev.type == "cpu"):
            pred = model(x, y)
        loss = ((pred.float() - y) ** 2).mean()
        if not math.isfinite(loss.item()):
            log["status"] = f"diverged at step {step}"
            break
        opt.zero_grad()
        loss.backward()
        if a.clip:
            torch.nn.utils.clip_grad_norm_(model.parameters(), a.clip)
        opt.step()
        losses.append(loss.item())
    save()
    torch.save(dict(model={k: v.cpu() for k, v in model.state_dict().items()}, config=vars(a)),
               a.out.replace(".json", ".pt"))


if __name__ == "__main__":
    main()
