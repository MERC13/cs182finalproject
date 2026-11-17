"""Is noise's effect on out-of-distribution error learning order or parameter jitter? (one-layer testbed)

  python scripts/ema_test_linattn.py runs/linattn/ema_test.json --procs 4

Re-runs the selected cells of the batch-size sweeps (Adam, Muon, exact Muon; λ ∝ 1/i² and 1/i, aligned;
batch 16-1,024; seeds 0-2; the same LRs) while logging a running average of Γ (EMA 0.95; and 0.99 at batch 16).
The predictor is linear in Γ, so the average removes step-to-step jitter but keeps the learning path.
"""
import argparse, json, os, sys, time
from multiprocessing import Pool
import torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from icl.linattn import run  # noqa: E402

SRC = {"1/i2-ali": ["runs/linattn/noise_sweep.json", "runs/linattn/noise_sweep_muonvar_exact.json"],
       "1/i-ali": ["runs/linattn/noise_sweep_1i.json"]}
OPTS = ["adam", "muon", "muon_exact"]


def job(a):
    cond, opt, lr, batch, seed, ema = a
    torch.set_num_threads(1)
    r = run(cond, opt, lr, mode="sep", steps=4000, batch=batch, seed=seed, log_every=10, ema=ema)
    r["batch"] = batch
    return r


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("out"); ap.add_argument("--procs", type=int, default=4)
    a = ap.parse_args(); t0 = time.time()
    jobs = []
    for cond, files in SRC.items():
        lr = {}
        for f in files:
            for r in json.load(open(f)):
                if r.get("selected") and r["cond"] == cond and r["opt"] in OPTS and r.get("muon_momentum") in (None, 0.95):
                    lr[(r["opt"], r["batch"])] = r["lr"]
        for (o, b), l in sorted(lr.items()):
            for s in (0, 1, 2):
                jobs.append((cond, o, l, b, s, 0.95))
                if b == 16:
                    jobs.append((cond, o, l, b, s, 0.99))
    with Pool(a.procs) as pool:
        res = pool.map(job, jobs, chunksize=1)
    json.dump(res, open(a.out, "w"))
    print(f"{len(res)} runs in {time.time() - t0:.0f}s -> {a.out}")


if __name__ == "__main__":
    main()
