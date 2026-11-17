"""Re-run the selected batch-1,024 linear-attention runs with the full learned matrix logged.

  python scripts/ood_rerun_linattn.py runs/linattn/ood_gamma.json --procs 4

Arms: SGD, Adam, Muon (Newton-Schulz), and the spectral knob p = 1, 0.5, 0.25, 0.1, 0 (p = 0 = exact Muon), at each
arm's selected learning rate, seeds 0-2, lambda ∝ 1/i^2 and 1/i, aligned. The runs are deterministic, so the
trajectories are the ones already analysed; only Gamma is added to the log (scripts/ood_map_linattn.py uses it).
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
from icl.linattn import run  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FILES = {"1/i2-ali": ["runs/linattn/noise_sweep.json", "runs/linattn/noise_sweep_muonvar_exact.json",
                      "runs/linattn/schatten_pilot.json"],
         "1/i-ali": ["runs/linattn/noise_sweep_1i.json", "runs/linattn/schatten_pilot.json"]}
ARMS = {"gd", "adam", "muon", "muon_exact", "muon_pow1", "muon_pow0.5", "muon_pow0.25", "muon_pow0.1"}


def job(args):
    cond, opt, lr, seed = args
    torch.set_num_threads(1)
    r = run(cond, opt, lr, mode="sep", steps=4000, batch=1024, seed=seed, log_every=10, muon_momentum=0.95,
            log_gamma=True)
    r["batch"] = 1024
    r["selected"] = True
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--procs", type=int, default=4)
    a = ap.parse_args()
    cells = {}
    for cond, files in FILES.items():
        for f in files:
            for r in json.load(open(os.path.join(ROOT, f))):
                if (r.get("selected") and r["cond"] == cond and r["batch"] == 1024 and r["opt"] in ARMS
                        and r.get("muon_momentum") in (None, 0.95)):
                    cells[(cond, r["opt"])] = r["lr"]
    jobs = [(c, o, lr, s) for (c, o), lr in sorted(cells.items()) for s in (0, 1, 2)]
    t0 = time.time()
    with Pool(a.procs) as pool:
        res = pool.map(job, jobs, chunksize=1)
    json.dump(res, open(a.out, "w"))
    print(f"{len(res)} runs in {time.time() - t0:.0f}s -> {a.out}")


if __name__ == "__main__":
    main()
