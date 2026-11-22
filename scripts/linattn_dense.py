"""Re-run the selected linear-attention arms (lambda ~ 1/i^2, aligned, batch 1,024) with the population loss and the
gain profile logged at EVERY step, for the matched-level PCR-vs-ridge comparison (scripts/analyze_linattn_matched.py).

  python3 scripts/linattn_dense.py runs/linattn/dense_1i2.json --procs 4

Same arms, learning rates (selected on seed 0 by scripts/noise_sweep_linattn.py and scripts/schatten_pilot_linattn.py)
and seeds 0-2 as the original runs. Training is deterministic given the seed and logging uses no randomness, so the
trajectories are the original ones; only the logging is denser (checked below against the original 10-step logs).
Runs last 6,000 instead of 4,000 steps so that every SGD seed reaches excess loss 0.1; the learning rate is constant,
so the first 4,000 steps are unchanged.
"""
import argparse
import json
import os
import sys
import time
from multiprocessing import Pool

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from icl.linattn import run  # noqa: E402

COND, BATCH, STEPS = "1/i2-ali", 1024, 6000
ORIG = ["runs/linattn/noise_sweep.json", "runs/linattn/noise_sweep_muonvar_exact.json", "runs/linattn/schatten_pilot.json"]
ARMS = ["gd", "adam", "muon_pow1", "muon_pow0.75", "muon_pow0.5", "muon_pow0.25", "muon_pow0.1", "muon_exact", "muon"]


def originals():
    out = {}
    for f in ORIG:
        for r in json.load(open(os.path.join(ROOT, f))):
            if (r.get("selected") and r["cond"] == COND and r.get("batch", BATCH) == BATCH
                    and r.get("muon_momentum") in (None, 0.95) and r["opt"] in ARMS):
                out[r["opt"], r["seed"]] = r
    return out


def job(args):
    opt, lr, seed = args
    torch.set_num_threads(1)
    r = run(COND, opt, lr, mode="sep", steps=STEPS, batch=BATCH, seed=seed, log_every=1, muon_momentum=0.95)
    r["batch"] = BATCH
    r["selected"] = True
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--procs", type=int, default=4)
    a = ap.parse_args()
    orig = originals()
    jobs = sorted({(o, orig[o, s]["lr"], s) for (o, s) in orig}, key=lambda x: (ARMS.index(x[0]), x[2]))
    print(f"{len(jobs)} runs:", [(o, lr, s) for o, lr, s in jobs], flush=True)
    t0 = time.time()
    with Pool(a.procs) as pool:
        res = pool.map(job, jobs, chunksize=1)
    print(f"done in {time.time() - t0:.0f}s", flush=True)
    worst = 0.0
    for r in res:
        o = orig[r["opt"], r["seed"]]
        st = np.array(r["steps"])
        for k, s in enumerate(o["steps"]):
            if s > STEPS:
                break
            i = int(np.searchsorted(st, s))
            worst = max(worst, abs(r["loss"][i] - o["loss"][k]) / max(abs(o["loss"][k]), 1e-12))
    print(f"largest relative difference from the original 10-step logs: {worst:.2e}", flush=True)
    json.dump(res, open(os.path.join(ROOT, a.out), "w"))


if __name__ == "__main__":
    main()
