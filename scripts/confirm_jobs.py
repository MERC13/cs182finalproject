"""Write the confirmation job lists (stable phase, then LR-decay branches resumed from checkpoints).

Protocol, fixed before any confirmation run:
  * fresh seeds 10, 11, 12 (model init and data stream); held-out evaluation prompts (seed 30000)
  * per-condition tuned configurations (best AUC in the search); 1/i^2 rotated uses the 1/i^2 aligned config
  * dense evaluation early (every 25 steps to step 2000), implied weights every 250 steps
  * stable constant-LR phase, checkpoint, then a linear decay-to-zero branch for converged end states

  python scripts/confirm_jobs.py --adamw iso=LR:B2,ani=LR:B2 --muon iso=LR,ani=LR [--stage stable|decay]
"""
import argparse

COMMON = ("--y_mode sep --readin_init gauss --eval_seed 30000 --eval_every 100 --dense_until 2000 "
          "--dense_every 25 --qshift_every 500 --iw_every 250 --save_model")
STEPS = {"adamw": (8000, 2000), "muon": (3000, 1000)}  # (stable steps, decay steps)
SEEDS = {"iso": (10, 11, 12), "1/i2-ali": (10, 11, 12), "1/i2-rot": (10, 11, 12)}
MUON_ROT_SEEDS = (10, 11)


def parse_cfg(s):
    out = {}
    for kv in s.split(","):
        k, v = kv.split("=")
        out[k] = v.split(":")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adamw", required=True)
    ap.add_argument("--muon", required=True)
    ap.add_argument("--stage", default="stable", choices=["stable", "decay"])
    a = ap.parse_args()
    cfg = {"adamw": parse_cfg(a.adamw), "muon": parse_cfg(a.muon)}
    jobs = []
    for cond in ("iso", "1/i2-ali", "1/i2-rot"):
        regime = "iso" if cond == "iso" else "ani"
        for opt in ("adamw", "muon"):
            seeds = MUON_ROT_SEEDS if (opt == "muon" and cond == "1/i2-rot") else SEEDS[cond]
            c = cfg[opt][regime]
            hp = f"--lr {c[0]}" + (f" --beta2 {c[1]}" if opt == "adamw" and len(c) > 1 else "")
            stable, decay = STEPS[opt]
            for s in seeds:
                tag = f"{opt}_{cond.replace('/', '')}_s{s}"
                base = f"runs/confirm/{tag}"
                if a.stage == "stable":
                    jobs.append((stable, f"--opt {opt} {hp} --cond {cond} --steps {stable} --seed {s} {COMMON} --out {base}.json"))
                else:
                    jobs.append((decay, f"--opt {opt} {hp} --cond {cond} --steps {stable + decay} --seed {s} {COMMON} "
                                        f"--resume {base}.ckpt --schedule decay --decay_steps {decay} --out {base}_decay.json"))
    for _, j in sorted(jobs, key=lambda t: -t[0]):  # longest first
        print(j)


if __name__ == "__main__":
    main()
