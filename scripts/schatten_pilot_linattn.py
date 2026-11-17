"""Spectral-flattening knob in the one-layer linear-attention testbed (separate rank-1 K/Q, aligned covariance).

  python scripts/schatten_pilot_linattn.py runs/linattn/schatten_pilot.json --procs 4

Muon's update with its singular values raised to a power p (icl.optim.spectral_power): p = 1 keeps the
gradient's own spectrum (per-matrix normalised momentum SGD), p = 0 is the exact polar factor (Muon with
exact orthogonalisation, already in runs/linattn/noise_sweep*.json). Everything else is shared: Nesterov
momentum 0.95, the Frobenius size of each step, Adam on the per-head scalars. In between, the update is
steepest descent in the Schatten-q norm with q = 1 + 1/p.

Same protocol as noise_sweep_linattn.py: batch 1,024, 4,000 steps, a 6-point LR sweep on seed 0 selected
by mean loss over training, then seeds 1-2 at the selected LR.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from multiprocessing import Pool

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from icl.linattn import run, score  # noqa: E402

CONDS = ["1/i2-ali", "1/i-ali"]
POWERS = [1.0, 0.75, 0.5, 0.25, 0.1]
LRS = [1e-3, 3e-3, 1e-2, 2e-2, 5e-2, 1e-1]
BATCH, STEPS = 1024, 4000


def job(args):
    cond, opt, lr, seed = args
    torch.set_num_threads(1)
    r = run(cond, opt, lr, mode="sep", steps=STEPS, batch=BATCH, seed=seed, log_every=10, muon_momentum=0.95)
    r["batch"] = BATCH
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--procs", type=int, default=4)
    a = ap.parse_args()
    t0 = time.time()
    cells = [(c, f"muon_pow{p:g}") for c in CONDS for p in POWERS]
    with Pool(a.procs) as pool:
        sweep = pool.map(job, [(c, o, lr, 0) for c, o in cells for lr in LRS], chunksize=1)
        print(f"[{time.time() - t0:5.0f}s] LR sweep done ({len(sweep)} runs)", flush=True)
        best = {}
        for r in sweep:
            r["sweep"] = True
            k = (r["cond"], r["opt"])
            if k not in best or score(r) < score(best[k]):
                best[k] = r
        for r in best.values():
            r["selected"] = True
        seeds = pool.map(job, [(c, o, best[(c, o)]["lr"], s) for c, o in cells for s in (1, 2)], chunksize=1)
        for r in seeds:
            r["selected"] = True
    json.dump(sweep + seeds, open(a.out, "w"))
    for c, o in cells:
        print(f"{c:9s} {o:12s}: lr {best[(c, o)]['lr']:g}  final loss {best[(c, o)]['loss'][-1]:.4f}")
    print(f"done in {time.time() - t0:.0f}s -> {a.out}")


if __name__ == "__main__":
    main()
