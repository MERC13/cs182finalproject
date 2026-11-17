"""Model ladder: multi-layer linear transformers between the one-layer testbed and the GPT-2.

  python scripts/ladder_sweep.py runs/ladder/ladder.json --procs 2

Arms: L = 1, 2, 4 on raw concatenated (x, y) tokens, and L = 2 with a learned 32-wide read-in (icl/lintf.py).
Per arm: SGD (momentum 0.9, clip 1), Adam, Muon x λ ∝ 1/i² aligned / rotated; a 3-point LR sweep on seed 0
(selection: mean loss over training), then seed 1 at the selected LR. Batch 256, 2,000 steps.
"""
import argparse, json, os, sys, time
from multiprocessing import Pool
import torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from icl.lintf import run  # noqa: E402

ARMS = [dict(L=1, readin="none"), dict(L=2, readin="none"), dict(L=4, readin="none"), dict(L=2, readin="learned", D=32)]
CONDS = ["1/i2-ali", "1/i2-rot"]
LRS = {"sgd": [0.03, 0.1, 0.3], "adam": [3e-3, 1e-2, 3e-2], "muon": [1e-3, 3e-3, 1e-2]}
STEPS = 2000


def job(a):
    arm, cond, opt, lr, seed = a
    torch.set_num_threads(1)
    r = run(cond, opt, lr, steps=STEPS, seed=seed, log_every=100, n_eval=1024, **arm)
    r["arm"] = f"L{arm['L']}-{arm['readin']}"
    return r


def score(r):
    return sum(r["loss"]) / len(r["loss"]) if r["status"] == "done" else float("inf")


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("out"); ap.add_argument("--procs", type=int, default=2)
    a = ap.parse_args(); t0 = time.time()
    cells = [(arm, c, o) for arm in ARMS for c in CONDS for o in LRS]
    with Pool(a.procs) as pool:
        sweep = pool.map(job, [(arm, c, o, lr, 0) for arm, c, o in cells for lr in LRS[o]], chunksize=1)
        print(f"[{time.time() - t0:5.0f}s] sweep done", flush=True)
        best = {}
        for r in sweep:
            r["sweep"] = True
            k = (r["arm"], r["cond"], r["opt"])
            if k not in best or score(r) < score(best[k]):
                best[k] = r
        for r in best.values():
            r["selected"] = True
        key = lambda arm: f"L{arm['L']}-{arm['readin']}"
        seeds = pool.map(job, [(arm, c, o, best[(key(arm), c, o)]["lr"], 1) for arm, c, o in cells], chunksize=1)
        for r in seeds:
            r["selected"] = True
    json.dump(sweep + seeds, open(a.out, "w"))
    for (arm, c, o), r in sorted(best.items()):
        print(f"{arm:12s} {c:9s} {o:5s} lr {r['lr']:g}  final loss {r['loss'][-1]:.4f}")
    print(f"done in {time.time() - t0:.0f}s -> {a.out}")


if __name__ == "__main__":
    main()
