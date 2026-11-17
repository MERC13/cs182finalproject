"""Query shifts both ways in the redesigned GPT-2 (jobs/ood_gpt2.txt), at matched in-distribution error.

  python scripts/analyze_ood_gpt2.py

The query spectrum is lam^(1-2t), same trace: t < 0 puts more query variance on the axes that had most variance in
training, t > 0 on the axes that had least (t = 1: inverted). Errors are relative (sum (yhat - y)^2 / sum y^2 over 256
prompts with the same contexts and tasks), read at the first step where the in-distribution query error (t = 0) on
those prompts falls below each level, by log-linear interpolation between evaluations (every 50 steps).
"""
import json
import os

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUNS = [("SGD", "runs/ood/sgd_s10.json"), ("AdamW", "runs/ood/adamw_s10.json"), ("Muon", "runs/ood/muon_s10.json"),
        ("Muon, exact", "runs/ood/muon_exact_s10.json")]
LEVELS = (0.05, 0.03, 0.01)


def matched(steps_q, lv, i0):
    """Each shift's error at the first crossing of error lv at t = 0 (log-linear interpolation)."""
    for (a, b) in zip(steps_q[:-1], steps_q[1:]):
        if a[i0] > lv >= b[i0] and a[i0] > 0 and b[i0] > 0:
            w = (np.log(a[i0]) - np.log(lv)) / (np.log(a[i0]) - np.log(b[i0]))
            return np.exp(np.log(a) + w * (np.log(b) - np.log(a)))
    return None


def main():
    res = {}
    for name, f in RUNS:
        path = os.path.join(ROOT, f)
        if not os.path.exists(path):
            continue
        d = json.load(open(path))
        ts = [float(t) for t in d["config"]["qshift_ts"].split(",")]
        Q = [np.array(e["qshift_err"]) for e in d["evals"] if "qshift_err" in e]
        i0 = ts.index(0.0)
        res[name] = dict(ts=ts, status=d.get("status"), at={lv: matched(Q, lv, i0) for lv in LEVELS})
    if not res:
        print("no runs yet")
        return
    ts = next(iter(res.values()))["ts"]
    order = sorted(range(len(ts)), key=lambda k: ts[k])
    for lv in LEVELS:
        print(f"\n-- matched in-distribution query error {lv}: shifted-query error by t "
              f"(< 0: toward the high-variance axes, > 0: toward the low-variance axes); ratio to AdamW in brackets")
        print(f"{'optimizer':12s} {'status':8s} " + " ".join(f"{'t = ' + format(ts[k], 'g'):>15s}" for k in order))
        base = res.get("AdamW", {}).get("at", {}).get(lv)
        for name, v in res.items():
            q = v["at"][lv]
            if q is None:
                print(f"{name:12s} {v['status']:8s} not reached")
                continue
            cells = [f"{q[k]:7.3f} ({q[k] / base[k]:4.2f})" if base is not None else f"{q[k]:7.3f}" for k in order]
            print(f"{name:12s} {v['status']:8s} " + " ".join(f"{c:>15s}" for c in cells))
    out = {n: {str(lv): (None if v["at"][lv] is None else v["at"][lv].tolist()) for lv in LEVELS} | {"ts": v["ts"]}
           for n, v in res.items()}
    json.dump(out, open(os.path.join(ROOT, "figures/noise/ood_gpt2.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
