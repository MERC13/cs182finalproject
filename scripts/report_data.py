"""Data behind the report's figures (Garg-scale runs, seeds 10-12). Writes figures/report/*.json.

  python scripts/report_data.py

curves.json  training loss and validation loss (squared error / d averaged over all 41 positions, on the 1,000
             held-out prompts) per evaluation step: mean and sample standard deviation over the seeds that
             reached that step (n given per row)
h1.json      weak-direction fraction learned (1 - probe error, mean over directions 3-20) at the first
             evaluation with error at 40 examples <= level: mean, sd, per-seed values
profile.json fraction learned for every direction at matched error 0.04: mean and sd over seeds
h2.json      shifted-query error / AdamW's (same seed) at matched t = 0 error, per t: mean, sd, per-seed values
speed.json   first step with error at 40 examples <= 0.05, per seed
"""
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
from icl.analysis import load  # noqa: E402
from analyze_garg_scale import q_matched  # noqa: E402

DIR = os.path.join(ROOT, "runs/garg_scale")
OUT = os.path.join(ROOT, "figures/report")
SEEDS = (10, 11, 12)
ARMS = [("SGD", "sgd_s{}.json"), ("AdamW", "adamw_s{}.json"), ("Muon", "muon_s{}.json"),
        ("Muon, p = 1", "muon_pow1_lr0.0005_s{}.json")]
LEVELS = (0.1, 0.05, 0.04, 0.03)


def stats(v):
    v = [x for x in v if x is not None]
    if not v:
        return None, None, 0
    return float(np.mean(v)), (float(np.std(v, ddof=1)) if len(v) > 1 else None), len(v)


def main():
    os.makedirs(OUT, exist_ok=True)
    runs = {(a, s): (load(os.path.join(DIR, f.format(s))), json.load(open(os.path.join(DIR, f.format(s)))))
            for a, f in ARMS for s in SEEDS}

    curves = []
    for a, _ in ARMS:
        per = {}
        for s in SEEDS:
            d = runs[a, s][1]
            for e in d["evals"]:
                if e.get("train_loss") is None:
                    continue
                per.setdefault(e["step"], []).append((e["train_loss"], float(np.mean(e["test_err_k"]))))
        for step in sorted(per):
            tr, va = zip(*per[step])
            mt, st, n = stats(tr)
            mv, sv, _ = stats(va)
            curves.append(dict(arm=a, step=step, train=mt, train_sd=st, val=mv, val_sd=sv, n=n))
    json.dump(curves, open(os.path.join(OUT, "curves.json"), "w"), indent=0)

    h1, h2, profile, speed = [], [], [], []
    for lv in LEVELS:
        weak = {}
        for (a, s), (r, _) in runs.items():
            i = np.nonzero(r.err_last <= lv)[0]
            weak[a, s] = float(np.mean(1 - r.probe_err()[int(i[0])][2:])) if len(i) else None
            if lv == 0.04 and len(i):
                profile.append(dict(arm=a, seed=s, frac=(1 - r.probe_err()[int(i[0])]).round(4).tolist()))
        for a, _ in ARMS:
            vals = [weak[a, s] for s in SEEDS]
            m, sd, n = stats(vals)
            h1.append(dict(arm=a, level=lv, mean=m, sd=sd, n=n, seeds=vals))
        for s in SEEDS:
            ts, base = q_matched(runs["AdamW", s][1], lv)
            if base is None:
                continue
            for a, _ in ARMS:
                if a == "AdamW":
                    continue
                _, q = q_matched(runs[a, s][1], lv)
                if q is None:
                    continue
                for t, x, b in zip(ts, q, base):
                    if t != 0:
                        h2.append(dict(level=lv, arm=a, seed=s, t=t, ratio=float(x / b)))
    h2s = []
    for lv in LEVELS:
        for a, _ in ARMS:
            for t in sorted({r["t"] for r in h2}):
                v = [r["ratio"] for r in h2 if r["level"] == lv and r["arm"] == a and r["t"] == t]
                if v:
                    m, sd, n = stats(v)
                    h2s.append(dict(level=lv, arm=a, t=t, mean=m, sd=sd, n=n, seeds=v))
    prof = []
    for a, _ in ARMS:
        P = np.array([p["frac"] for p in profile if p["arm"] == a])
        if len(P):
            for j in range(P.shape[1]):
                prof.append(dict(arm=a, direction=j + 1, mean=float(P[:, j].mean()),
                                 sd=float(P[:, j].std(ddof=1)) if len(P) > 1 else None, n=len(P)))
    for a, _ in ARMS:
        for s in SEEDS:
            r = runs[a, s][0]
            i = np.nonzero(r.err_last <= 0.05)[0]
            speed.append(dict(arm=a, seed=s, step=int(r.steps[i[0]]) if len(i) else None))
    json.dump(h1, open(os.path.join(OUT, "h1.json"), "w"), indent=0)
    json.dump(h2s, open(os.path.join(OUT, "h2.json"), "w"), indent=0)
    json.dump(prof, open(os.path.join(OUT, "profile.json"), "w"), indent=0)
    json.dump(speed, open(os.path.join(OUT, "speed.json"), "w"), indent=0)
    print(f"{len(curves)} curve rows, {len(h1)} H1 rows, {len(h2s)} H2 rows, {len(prof)} profile rows -> {OUT}")


if __name__ == "__main__":
    main()
