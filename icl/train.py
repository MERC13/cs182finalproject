"""Train one model and log learning curves, probes, estimator distances, implied weights and
the query-shift test.  Usage:

  python -m icl.train --opt muon --lr 0.005 --cond 1/i2-ali --steps 6000 --seed 0 --out runs/x.json

Resume / branch (e.g. a learning-rate-decay branch from a constant-LR checkpoint):

  python -m icl.train ... --resume runs/x.ckpt --steps 7500 --schedule decay --decay_steps 1500 --out runs/x_decay.json
"""
from __future__ import annotations

import argparse
import json
import math
import os
import time

import numpy as np
import torch

from .data import Sampler, make_condition, make_eval_sets, make_pool, zipf_probs
from .estimators import Reference
from .model import GPT2ICL
from .optim import build_optimizer
from .transience import make_tr_sets, reference_errors, tr_eval

VAL_SEED, TEST_SEED, HELDOUT_SEED = 10_000, 20_000, 30_000


def parse(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--opt", required=True, choices=["sgd", "adamw", "muon"])
    p.add_argument("--lr", type=float, required=True)
    p.add_argument("--aux_lr", type=float, default=1e-3, help="AdamW lr for Muon's non-hidden params")
    p.add_argument("--muon_params", default="hidden", choices=["hidden", "hidden+in"])
    p.add_argument("--muon_orth", default="ns", choices=["ns", "exact", "fro", "none", "pow"],
                   help="fro: per-matrix normalised SGD ablation; none: SGD on the hidden matrices; "
                        "pow: singular values of the update raised to --muon_power (1 = fro, 0 = exact)")
    p.add_argument("--muon_power", type=float, default=0.0, help="spectral power for --muon_orth pow")
    p.add_argument("--muon_momentum", type=float, default=0.95)
    p.add_argument("--momentum", type=float, default=0.9, help="SGD momentum")
    p.add_argument("--nesterov", type=int, default=0, help="SGD Nesterov momentum")
    p.add_argument("--beta1", type=float, default=0.9, help="AdamW (and Muon's aux AdamW) beta1")
    p.add_argument("--beta2", type=float, default=0.999, help="AdamW (and Muon's aux AdamW) beta2")
    p.add_argument("--wd", type=float, default=0.0)
    p.add_argument("--clip", type=float, default=0.0, help="grad-norm clip (0 = off)")
    p.add_argument("--param_rotation", type=int, default=0, help="AdamW in a randomly rotated parameter space (Xie et al.)")
    p.add_argument("--cond", default="iso")
    p.add_argument("--d", type=int, default=8)
    p.add_argument("--n_points", type=int, default=33)
    p.add_argument("--n_embd", type=int, default=128)
    p.add_argument("--n_layer", type=int, default=4)
    p.add_argument("--n_head", type=int, default=4)
    p.add_argument("--y_mode", default="garg", choices=["garg", "sep", "concat"])
    p.add_argument("--readin_init", default="torch", choices=["torch", "gauss", "eye"])
    p.add_argument("--hidden_init", type=float, default=0.0, help="std for all hidden matrices (0 = GPT-2 init)")
    p.add_argument("--freeze_readin", type=int, default=0,
                   help="1: keep the read-in weight fixed at its init (e.g. eye: the input axes stay the residual axes)")
    p.add_argument("--n_tasks", type=int, default=0, help="finite training-task pool of this size (0 = fresh w)")
    p.add_argument("--noise", type=float, default=0.0, help="label noise std in training (and transience evals)")
    p.add_argument("--pool_seed", type=int, default=4242)
    p.add_argument("--zipf", type=float, default=0.0, help="task frequencies ∝ (k+1)^-zipf over the pool (0 = uniform)")
    p.add_argument("--n_tr", type=int, default=512, help="prompts per transience evaluation set")
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--steps", type=int, default=6000)
    p.add_argument("--warmup", type=int, default=100)
    p.add_argument("--schedule", default="constant", choices=["constant", "decay"],
                   help="decay: LR falls linearly to 0 over the last --decay_steps steps")
    p.add_argument("--decay_steps", type=int, default=0)
    p.add_argument("--curriculum", default="none", choices=["none", "garg", "points"],
                   help="garg: active dims cur_dims0 -> d (+1 per interval) with 4*dims+1 points; points: points only")
    p.add_argument("--cur_interval", type=int, default=1000)
    p.add_argument("--cur_dims0", type=int, default=2)
    p.add_argument("--cur_points0", type=int, default=9)
    p.add_argument("--eval_every", type=int, default=100)
    p.add_argument("--dense_until", type=int, default=0, help="also evaluate every --dense_every steps up to here")
    p.add_argument("--dense_every", type=int, default=25)
    p.add_argument("--qshift_every", type=int, default=500)
    p.add_argument("--qshift_ts", default="0,0.25,0.5,0.75,1",
                   help="query-shift levels t (query spectrum lam^(1-2t)); keep 0 first and 1 last for the standard "
                        "analyses; t < 0 moves query variance onto the axes that had most variance in training")
    p.add_argument("--iw_every", type=int, default=0, help="implied-weight fit every N steps (0 = with qshift)")
    p.add_argument("--n_test", type=int, default=1000)
    p.add_argument("--eval_seed", type=int, default=TEST_SEED, help="prompt seed for periodic evaluation")
    p.add_argument("--light_eval", action="store_true", help="skip probes/estimators during training")
    p.add_argument("--prune_step", type=int, default=0, help="stop early if still on the plateau at this step")
    p.add_argument("--prune_err", type=float, default=0.97)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--threads", type=int, default=1)
    p.add_argument("--bf16", type=int, default=1, help="bf16 autocast for linear layers in training on CPU (eval is fp32)")
    p.add_argument("--device", default="cpu", help="cpu or cuda (data is sampled on CPU and moved)")
    p.add_argument("--out", required=True)
    p.add_argument("--save_model", action="store_true")
    p.add_argument("--save_ckpt", type=int, default=1, help="save a resumable checkpoint at the end")
    p.add_argument("--ckpt_every", type=int, default=0)
    p.add_argument("--resume", default="", help="checkpoint to continue from")
    return p.parse_args(argv)


@torch.no_grad()
def predict(model, x, y, chunk=500):
    """Model predictions for (CPU) prompt tensors; returns CPU float32 predictions."""
    dev = next(model.parameters()).device
    return torch.cat([model(x[i:i + chunk].to(dev), y[i:i + chunk].to(dev)).float().cpu()
                      for i in range(0, len(x), chunk)])


def r(a, nd=5):
    a = np.asarray(a, dtype=np.float64)
    return np.vectorize(lambda v: float(f"{v:.{nd}g}"))(a).tolist() if a.ndim else float(f"{a:.{nd}g}")


@torch.no_grad()
def implied_weights(model, ev, cond):
    """Fit yhat(x_q) = w_hat . x_q + b per context; report per-principal-axis gain and error."""
    Pi, Q, d = ev.iq.shape
    yhat = predict(model, ev.ix, ev.iy)[:, -1].reshape(Pi, Q).double()
    Xq = torch.cat([ev.iq.double(), torch.ones(Pi, Q, 1, dtype=torch.float64)], -1)
    coef = torch.linalg.lstsq(Xq, yhat.unsqueeze(-1)).solution.squeeze(-1)  # (Pi, d+1)
    resid = yhat - (Xq @ coef.unsqueeze(-1)).squeeze(-1)
    r2 = 1 - (resid ** 2).sum() / ((yhat - yhat.mean(1, keepdim=True)) ** 2).sum().clamp_min(1e-12)
    U = torch.tensor(cond.U, dtype=torch.float64)
    a = coef[:, :d] @ U  # implied weights in the principal-axis basis
    b = ev.iw.double() @ U  # true weights in that basis
    gain = (a * b).sum(0) / (b * b).sum(0)
    err = ((a - b) ** 2).sum(0) / (b * b).sum(0)
    return {"iw_gain": r(gain), "iw_err": r(err), "iw_r2": r(r2), "iw_bias_rms": r(coef[:, d].pow(2).mean().sqrt())}


@torch.no_grad()
def evaluate(model, ev, ref, cond, full=True, qshift=False, iw=False):
    d = cond.d
    model.eval()
    out = {}
    pred = predict(model, ev.x, ev.y)
    err_k = ((pred - ev.y) ** 2).mean(0) / d  # (n,)
    out["test_err_k"] = r(err_k)
    if full:
        D = ref.distances(pred)  # (E, n)
        out["dist_1_32"] = dict(zip(ref.names, r(D[:, 1:].mean(1))))
        out["dist_8_32"] = dict(zip(ref.names, r(D[:, 8:].mean(1))))
        pp = predict(model, ev.px, ev.py)
        # relative error sum (pred - y)^2 / sum y^2 per axis and prefix: exactly 1 for a zero predictor
        perr = torch.stack([((pp[ev.pdir == k] - ev.py[ev.pdir == k]) ** 2).sum(0)
                            / (ev.py[ev.pdir == k] ** 2).sum(0).clamp_min(1e-12) for k in range(d)])
        out["probe_err_k"] = r(perr)  # (d, n); 1 = axis not used
    if qshift or iw:
        out.update(implied_weights(model, ev, cond))
    if qshift:
        T, Pq, n, _ = ev.qx.shape
        qp = predict(model, ev.qx.reshape(T * Pq, n, d), ev.qy.reshape(T * Pq, n)).reshape(T, Pq, n)
        out["qshift_err"] = r(((qp[..., -1] - ev.qy[..., -1]) ** 2).sum(1) / (ev.qy[..., -1] ** 2).sum(1))
    model.train()
    return out


def lr_scale(a, step):
    s = min(1.0, (step + 1) / a.warmup) if a.warmup else 1.0
    if a.schedule == "decay" and a.decay_steps:
        start = a.steps - a.decay_steps
        if step >= start:
            s *= max(0.0, (a.steps - step) / a.decay_steps)
    return s


def curriculum(a, step):
    """(active dims, prompt length) at this step."""
    if a.curriculum == "none":
        return a.d, a.n_points
    k = step // a.cur_interval
    if a.curriculum == "garg":
        dc = min(a.d, a.cur_dims0 + k)
        return dc, min(a.n_points, 4 * dc + 1)
    return a.d, min(a.n_points, a.cur_points0 + 4 * k)


def eval_steps(a):
    s = {0, 25, 50} | set(range(a.eval_every, a.steps + 1, a.eval_every))
    if a.dense_until:
        s |= set(range(0, min(a.dense_until, a.steps) + 1, a.dense_every))
    return s


def main(argv=None):
    a = parse(argv)
    torch.set_num_threads(a.threads)
    torch.manual_seed(a.seed)
    cond = make_condition(a.cond, a.d)
    dev = torch.device(a.device)
    model = GPT2ICL(a.d, a.n_points, a.n_embd, a.n_layer, a.n_head, a.y_mode, a.readin_init, a.hidden_init).to(dev)
    if a.freeze_readin:
        model.read_in.weight.requires_grad_(False)  # optimizers skip parameters without a gradient
    amp = bool(a.bf16) and dev.type == "cpu"  # fp32 on GPU: the model is tiny, and T4/P100 lack fast bf16
    opt = build_optimizer(model, a.opt, a.lr, aux_lr=a.aux_lr, momentum=a.momentum, weight_decay=a.wd,
                          betas=(a.beta1, a.beta2), muon_params=a.muon_params, muon_orth=a.muon_orth,
                          muon_momentum=a.muon_momentum, nesterov=bool(a.nesterov),
                          param_rotation=bool(a.param_rotation), seed=a.seed, muon_power=a.muon_power)
    pool = make_pool(a.n_tasks, a.d, a.pool_seed) if a.n_tasks else None
    pool_p = zipf_probs(a.n_tasks, a.zipf) if a.n_tasks and a.zipf else None
    sampler = Sampler(cond, a.n_points, seed=a.seed + 777, pool=pool, noise=a.noise, pool_p=pool_p)
    trsets = make_tr_sets(cond, a.n_points, a.eval_seed + 1, pool, a.noise, a.n_tr, pool_p) if a.n_tasks else None
    ev = make_eval_sets(cond, a.n_points, a.eval_seed, n_test=a.n_test, qts=tuple(float(t) for t in a.qshift_ts.split(",")))
    ref = Reference(ev.x, ev.y, cond.lam, cond.U)
    ckpt_path = a.out.replace(".json", ".ckpt")
    start = 0
    if a.resume:
        ck = torch.load(a.resume, weights_only=False)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        sampler.g.set_state(ck["sampler"])
        torch.set_rng_state(ck["torch_rng"])
        log = ck["log"]
        log.setdefault("resumed", []).append(dict(from_ckpt=a.resume, at_step=ck["step"], config=vars(a)))
        log["status"] = "running"
        start = ck["step"]
        assert a.eval_seed == log["config"]["eval_seed"], "resume with the same evaluation prompts"
    else:
        log = dict(config=vars(a), cond=dict(name=cond.name, lam=cond.lam.tolist()),
                   ref_names=ref.names, ref_err_k=r(ref.errors()), evals=[], status="running")
        if trsets is not None:
            log["tr_ref_err_k"] = {k: r(v) for k, v in reference_errors(trsets, a.d).items()}
    for g in opt.param_groups:
        g.setdefault("base_lr", g["lr"])

    def save():
        tmp = a.out + ".tmp"
        with open(tmp, "w") as f:
            json.dump(log, f)
        os.replace(tmp, a.out)

    def save_ckpt(step):
        torch.save(dict(model={k: v.cpu() for k, v in model.state_dict().items()}, opt=opt.state_dict(),
                        sampler=sampler.g.get_state(), torch_rng=torch.get_rng_state(), step=step, log=log),
                   ckpt_path + ".tmp")
        os.replace(ckpt_path + ".tmp", ckpt_path)

    t0 = time.time() - (log["evals"][-1]["time"] if a.resume and log["evals"] else 0.0)
    losses, E = [], eval_steps(a)
    for step in range(start, a.steps + 1):
        if step in E and not (a.resume and step == start):
            dc, nc = curriculum(a, step)
            rec = dict(step=step, time=round(time.time() - t0, 2), lr_scale=round(lr_scale(a, step), 4),
                       train_loss=r(np.mean(losses) / a.d) if losses else None)
            if a.curriculum != "none":
                rec.update(cur_dims=dc, cur_points=nc)
            rec.update(evaluate(model, ev, ref, cond, full=not a.light_eval,
                                qshift=(step % a.qshift_every == 0) or step == a.steps,
                                iw=bool(a.iw_every) and step % a.iw_every == 0))
            if trsets is not None:
                rec.update({k: r(v) for k, v in tr_eval(predict, model, trsets, a.d).items()})
            log["evals"].append(rec)
            losses = []
            if step % (5 * a.eval_every) == 0:
                save()
            if a.prune_step and step == a.prune_step and np.mean(rec["test_err_k"][1:]) > a.prune_err:
                log["status"] = f"pruned@{step}"
                save()
                return log
        if a.ckpt_every and step % a.ckpt_every == 0 and step > start:
            save_ckpt(step)
        if step == a.steps:
            break
        s = lr_scale(a, step)
        for g in opt.param_groups:
            g["lr"] = g["base_lr"] * s
        dc, nc = curriculum(a, step)
        x, y = sampler.sample(a.batch, n_points=nc, d_active=dc)
        x, y = x.to(dev, non_blocking=True), y.to(dev, non_blocking=True)
        with torch.autocast(dev.type, dtype=torch.bfloat16, enabled=amp):
            pred = model(x, y)
        loss = ((pred.float() - y) ** 2).mean()
        if not math.isfinite(loss.item()) or loss.item() > 1e4 * a.d:
            log["status"] = f"diverged@{step}"
            save()
            return log
        opt.zero_grad(set_to_none=True)
        loss.backward()
        if a.clip:
            torch.nn.utils.clip_grad_norm_(model.parameters(), a.clip)
        opt.step()
        losses.append(loss.item())
    # final: validation prompts (seed separate from both evaluation sets) for LR selection
    evv = make_eval_sets(cond, a.n_points, VAL_SEED, n_test=a.n_test, n_probe=1, n_qshift=1)
    pv = predict(model, evv.x, evv.y)
    vk = ((pv - evv.y) ** 2).mean(0) / a.d
    log["final_val_err_k"] = r(vk)
    log["final_val_loss"] = r(vk.mean())
    log["status"] = "done"
    log["wall"] = round(time.time() - t0, 1)
    save()
    if a.save_ckpt:
        save_ckpt(a.steps)
    if a.save_model:
        torch.save({k: v.cpu() for k, v in model.state_dict().items()}, a.out.replace(".json", ".pt"))
    return log


if __name__ == "__main__":
    main()
