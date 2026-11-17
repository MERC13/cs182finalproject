"""Write job files for the LR sweep and the main grid.

  python scripts/make_jobs.py sweep  --steps 6000 > jobs/sweep.txt
  python scripts/make_jobs.py grid   --steps 8000 --lrs sgd=0.1,adamw=1e-3,muon=0.02 --aux 1e-3 > jobs/grid.txt
"""
import argparse

COMMON = "--y_mode sep --readin_init gauss"
SWEEP = {"sgd": [0.03, 0.1, 0.3], "adamw": [3e-4, 1e-3, 3e-3], "muon": [0.01, 0.02, 0.05]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("kind", choices=["sweep", "grid"])
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--conds", default="iso")
    ap.add_argument("--lrs", default="")
    ap.add_argument("--aux", type=float, default=1e-3)
    ap.add_argument("--seeds", default="0")
    ap.add_argument("--extra", default="")
    ap.add_argument("--prefix", default="")
    a = ap.parse_args()
    conds = a.conds.split(",")
    if a.kind == "sweep":
        for c in conds:
            for opt, lrs in SWEEP.items():
                for lr in lrs:
                    tag = f"{a.prefix}{opt}_lr{lr:g}_{c.replace('/', '')}"
                    print(f"--opt {opt} --lr {lr:g} --aux_lr {a.aux:g} --cond {c} --steps {a.steps} {COMMON} "
                          f"--light_eval --qshift_every 100000 {a.extra} --out runs/sweep/{tag}.json")
    else:
        lrs = dict(kv.split("=") for kv in a.lrs.split(","))
        for seed in a.seeds.split(","):
            for c in conds:
                for opt in ("sgd", "adamw", "muon"):
                    if opt not in lrs:
                        continue
                    tag = f"{a.prefix}{opt}_{c.replace('/', '')}_s{seed}"
                    print(f"--opt {opt} --lr {lrs[opt]} --aux_lr {a.aux:g} --cond {c} --steps {a.steps} --seed {seed} "
                          f"{COMMON} --save_model {a.extra} --out runs/grid/{tag}.json")


if __name__ == "__main__":
    main()
