"""Does minibatch noise gate the optimizer-geometry effects? One-layer linear attention, separate rank-1 K/Q.

  python scripts/noise_sweep_linattn.py runs/linattn/noise_sweep.json --procs 4

The main testbed grid used batch 1,024, where Adam pays 2-6.5x for a rotated covariance and Muon learns
all axes nearly together. The GPT-2 runs used batch 64 and showed neither effect. This sweep varies only
the batch size, to see whether noise alone removes them (cf. arXiv 2610.16868 for Muon).

Grid: λ ∝ 1/i², aligned and rotated × GD, Adam, SignGD, Muon (NS5) × batch 16, 64, 256, 1,024.
Each cell: a 6-point LR sweep on seed 0, selected by mean loss over training (as in the main grid),
then seeds 1-2 at the selected LR. 4,000 steps, exact population loss logged every 10 steps.
Muon variants: --opts muon,muon_exact --muon_momentum 0 (default momentum 0.95, Nesterov).
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

CONDS = ["1/i2-ali", "1/i2-rot"]
OPTS = ["gd", "adam", "signgd", "muon"]
BATCHES = [16, 64, 256, 1024]
LRS = {"gd": [0.003, 0.01, 0.02, 0.05, 0.1, 0.2], "adam": [3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1],
       "signgd": [1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2], "muon": [1e-3, 3e-3, 1e-2, 2e-2, 5e-2, 1e-1],
       "muon_exact": [1e-3, 3e-3, 1e-2, 2e-2, 5e-2, 1e-1]}
STEPS = 4000


def job(args):
    cond, opt, lr, batch, seed, mom = args
    torch.set_num_threads(1)
    r = run(cond, opt, lr, mode="sep", steps=STEPS, batch=batch, seed=seed, log_every=10, muon_momentum=mom)
    r["batch"] = batch
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--procs", type=int, default=4)
    ap.add_argument("--opts", default=",".join(OPTS))
    ap.add_argument("--conds", default=",".join(CONDS))
    ap.add_argument("--muon_momentum", type=float, default=0.95, help="Muon variants only (0 = no momentum)")
    a = ap.parse_args()
    t0 = time.time()
    mom = a.muon_momentum
    cells = [(c, o, b) for c in a.conds.split(",") for o in a.opts.split(",") for b in BATCHES]
    with Pool(a.procs) as pool:
        sweep = pool.map(job, [(c, o, lr, b, 0, mom) for c, o, b in cells for lr in LRS[o]], chunksize=1)
        print(f"[{time.time() - t0:5.0f}s] LR sweep done ({len(sweep)} runs)", flush=True)
        best = {}
        for r in sweep:
            r["sweep"] = True
            k = (r["cond"], r["opt"], r["batch"])
            if k not in best or score(r) < score(best[k]):
                best[k] = r
        for r in best.values():
            r["selected"] = True
        seeds = pool.map(job, [(c, o, best[(c, o, b)]["lr"], b, s, mom) for c, o, b in cells for s in (1, 2)],
                         chunksize=1)
        for r in seeds:
            r["selected"] = True
    json.dump(sweep + seeds, open(a.out, "w"))
    for c, o, b in cells:
        print(f"{c:9s} {o:7s} batch {b:5d}: lr {best[(c, o, b)]['lr']:g}  final loss {best[(c, o, b)]['loss'][-1]:.4f}")
    print(f"done in {time.time() - t0:.0f}s -> {a.out}")


if __name__ == "__main__":
    main()
