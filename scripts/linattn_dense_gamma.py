"""Re-run the selected linear-attention arms (lambda ~ 1/i^2, aligned, batch 1,024) logging the full learned matrix
Gamma at EVERY step, for the matched-level query-shift test (scripts/ood_map_linattn.py).

  python3 scripts/linattn_dense_gamma.py runs/linattn/ood_gamma_dense.json --procs 3

Same arms, learning rates and seeds (0-2) as runs/linattn/ood_gamma.json, which logged Gamma every 10 steps. Training
is deterministic given the seed and logging uses no randomness, so the trajectories are the same. Runs last 6,000
steps (constant learning rate, so the first 4,000 are unchanged). To keep the file small, only the steps from the
first one with in-distribution excess loss <= 0.25 to the first one with excess loss <= 0.03 are kept, which brackets
the matched levels 0.1 and 0.05.
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
SRC = "runs/linattn/ood_gamma.json"
HI, LO = 0.25, 0.03


def job(args):
    opt, lr, seed = args
    torch.set_num_threads(1)
    r = run(COND, opt, lr, mode="sep", steps=STEPS, batch=BATCH, seed=seed, log_every=1, muon_momentum=0.95,
            log_gamma=True)
    L = np.array(r["loss"])
    lam = np.array(r["lam"])
    Lstar = 1 - float(np.sum(np.array(r["rho_star"]) * lam)) / len(lam)
    exc = (L - Lstar) / (1 - Lstar)
    a = int(np.nonzero(exc <= HI)[0][0])
    lo = np.nonzero(exc <= LO)[0]
    b = int(lo[0]) + 1 if len(lo) else len(exc)
    for k in ("steps", "loss", "rho", "qloss", "gamma"):
        r[k] = r[k][a:b]
    r.update(batch=BATCH, selected=True, window=[HI, LO])
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--procs", type=int, default=3)
    a = ap.parse_args()
    src = [r for r in json.load(open(os.path.join(ROOT, SRC))) if r.get("selected") and r["cond"] == COND]
    jobs = sorted({(r["opt"], r["lr"], r["seed"]) for r in src})
    print(f"{len(jobs)} runs", flush=True)
    t0 = time.time()
    with Pool(a.procs) as pool:
        res = pool.map(job, jobs, chunksize=1)
    print(f"done in {time.time() - t0:.0f}s", flush=True)
    orig = {(r["opt"], r["seed"]): r for r in src}
    worst = 0.0
    for r in res:
        o = orig[r["opt"], r["seed"]]
        pos = {s: i for i, s in enumerate(r["steps"])}
        for k, s in enumerate(o["steps"]):
            if s in pos:
                worst = max(worst, float(np.abs(np.array(o["gamma"][k]) - np.array(r["gamma"][pos[s]])).max()))
    print(f"largest |Gamma| difference from the original 10-step logs: {worst:.2e}", flush=True)
    json.dump(res, open(os.path.join(ROOT, a.out), "w"))


if __name__ == "__main__":
    main()
