"""Learning-rate check at Garg scale (jobs/garg_lr.txt): is each arm's rate in the main runs the best of a x3 grid?

  python scripts/analyze_garg_lr.py

Seed 10, 1,500 steps (jobs/garg_lr.txt, plus AdamW 5e-4 in jobs/garg_lr_adamw5e-4.txt). Criterion: training loss averaged over steps 1,400-1,500 (the two logged 50-step averages at
steps 1,450 and 1,500). Every training prompt is new, so this estimates the population loss without touching the
evaluation prompts. The centre rates are read from the main runs (runs/garg_scale, seed 10) at the same steps;
their trajectories up to step 1,500 are the same as a 1,500-step run's (constant learning rate after warmup).
Also shown, for information only: error with 40 examples on the held-out prompts at step 1,500, the first step
with that error <= 0.1, and H1's probe score (raw and bounded, directions 3-20) at error 0.05 (first evaluation and
interpolated to exactly 0.05), if reached by step 1,500. The last checks that the H1 gap between Muon and AdamW does not
depend on either rate.
"""
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from icl.analysis import load  # noqa: E402

ARMS = [
    ("SGD", [("runs/garg_lr/sgd_lr0.03_s10.json", 0.03), ("runs/garg_scale/sgd_s10.json", 0.1),
             ("runs/garg_lr/sgd_lr0.3_s10.json", 0.3)]),
    ("AdamW", [("runs/garg_lr/adamw_lr1e-4_s10.json", 1e-4), ("runs/garg_scale/adamw_s10.json", 3e-4),
               ("runs/garg_lr/adamw_lr5e-4_s10.json", 5e-4),
               ("runs/garg_lr/adamw_lr1e-3_s10.json", 1e-3)]),
    ("Muon", [("runs/garg_lr/muon_lr0.0025_s10.json", 0.0025), ("runs/garg_scale/muon_s10.json", 0.007),
              ("runs/garg_lr/muon_lr0.02_s10.json", 0.02)]),
    ("Muon, p = 1", [("runs/garg_lr/muon_pow1_lr0.00015_s10.json", 1.5e-4),
                     ("runs/garg_scale/muon_pow1_lr0.0005_s10.json", 5e-4),
                     ("runs/garg_lr/muon_pow1_lr0.0015_s10.json", 1.5e-3),
                     ("runs/garg_scale/muon_pow1_lr0.007_s10.json", 7e-3)]),
]
USED = {"SGD": 0.1, "AdamW": 3e-4, "Muon": 0.007, "Muon, p = 1": 5e-4}
STEP = 1500
H1_LEVEL = 0.05


def row(path):
    p = os.path.join(ROOT, path)
    if not os.path.exists(p):
        return None
    d = json.load(open(p))
    ev = {e["step"]: e for e in d["evals"]}
    if STEP not in ev or (STEP - 50) not in ev:
        return dict(status=d.get("status"), last=d["evals"][-1]["step"] if d["evals"] else 0)
    tl = float(np.mean([ev[STEP]["train_loss"], ev[STEP - 50]["train_loss"]]))
    r = load(p)
    err = float(r.err_last[np.nonzero(r.steps == STEP)[0][0]])
    i = np.nonzero(r.err_last <= 0.1)[0]
    j = np.nonzero((r.err_last <= H1_LEVEL) & (r.steps <= STEP))[0]
    weak = float(np.mean(1 - r.probe_err()[int(j[0])][2:])) if len(j) else None
    out = dict(status=d.get("status"), train=tl, err=err, t01=int(r.steps[i[0]]) if len(i) else None, weak05=weak,
               t05=int(r.steps[j[0]]) if len(j) else None)
    if len(j):  # probe score (raw) and bounded score at error 0.05: first evaluation and interpolated to exactly 0.05
        for tag, interp in (("first", False), ("interp", True)):
            f = np.asarray(r.read_at(1 - r.probe_err(), H1_LEVEL, interp=interp))[2:]
            out[f"raw_{tag}"], out[f"bounded_{tag}"] = float(f.mean()), float(np.clip(f, 0, None).mean())
    return out


def main():
    out = {}
    print(f"{'arm':12s} {'lr':>8s} {'status':9s} {'train loss 1400-1500':>21s} {'error@40 (1500)':>16s} {'step to 0.1':>12s} {'H1 at 0.05':>11s}")
    for arm, grid in ARMS:
        res = [(lr, row(p)) for p, lr in grid]
        done = [(lr, x) for lr, x in res if x and "train" in x and np.isfinite(x["train"])]
        best = min(done, key=lambda t: t[1]["train"])[0] if done else None
        for lr, x in res:
            if x is None:
                print(f"{arm:12s} {lr:8.2g} {'missing':9s}")
                continue
            if "train" not in x:
                print(f"{arm:12s} {lr:8.2g} {str(x['status']):9s} (stopped at step {x['last']})")
                continue
            mark = (" best" if lr == best else "") + (" (used)" if lr == USED[arm] else "")
            print(f"{arm:12s} {lr:8.2g} {str(x['status']):9s} {x['train']:21.4f} {x['err']:16.4f} "
                  f"{str(x['t01']):>12s} {'-' if x['weak05'] is None else format(x['weak05'], '+.3f'):>11s}{mark}")
        out[arm] = dict(used=USED[arm], best=best, grid=[dict(lr=lr, **(x or {})) for lr, x in res])
    print("\nAt error 0.05 (runs that reach it by step 1,500): first step; probe score raw / bounded, at the first "
          "evaluation and interpolated to exactly 0.05")
    for arm in out:
        for g in out[arm]["grid"]:
            if g.get("t05") is not None:
                print(f"  {arm:12s} lr {g['lr']:<8g} step {g['t05']:5d}   first {g['raw_first']:+.3f} / {g['bounded_first']:.3f}"
                      f"   interpolated {g['raw_interp']:+.3f} / {g['bounded_interp']:.3f}")
    os.makedirs(os.path.join(ROOT, "figures/report"), exist_ok=True)
    json.dump(out, open(os.path.join(ROOT, "figures/report/garg_lr.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
