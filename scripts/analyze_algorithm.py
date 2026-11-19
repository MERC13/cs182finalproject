"""Which in-context algorithm does each optimizer's model run at matched error: closer to principal component
regression (PCR) or to ridge regression?  (Report: H1, second part; Table 4.)

  python scripts/analyze_algorithm.py

For each GPT-2 run, at the first evaluation whose error with the full prompt is at or below each matched level,
read the logged mean squared distance between the model's predictions and each reference estimator's on the
held-out prompts (`dist_8_32`: predictions with 8 or more examples; see icl/estimators.py for the references:
`pcr_m` is the Bayes version of PCR on the top m population directions, `ridge_l` is ridge with penalty l).
  ratio = (distance to the nearest PCR reference) / (distance to the nearest ridge reference)
  > 1: the model is closer to ridge than to any PCR;  < 1: closer to PCR.
Also printed: the best m and l, and the distance to least squares (OLS) for scale.
The linear-attention version of this question is the `ridge-fit` column of scripts/analyze_schatten_pilot.py.
"""
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from icl.analysis import load  # noqa: E402

SETTINGS = {
    "Garg-scale GPT-2 (d = 20)": dict(
        levels=(0.1, 0.05, 0.04, 0.03), seeds=(10, 11, 12),
        arms=[("SGD", "runs/garg_scale/sgd_s{}.json"), ("AdamW", "runs/garg_scale/adamw_s{}.json"),
              ("Muon", "runs/garg_scale/muon_s{}.json"), ("Muon, p = 1", "runs/garg_scale/muon_pow1_lr0.0005_s{}.json"),
              ("Muon, exact", "runs/garg_scale/muon_exact_s{}.json")]),
    "Small GPT-2 (d = 8)": dict(
        levels=(0.1, 0.05), seeds=(10, 11),
        arms=[("SGD", "runs/init/sgd_lr0.1_init044_s{}.json"), ("AdamW", "runs/init/adamw_init044_s{}.json"),
              ("Muon", "runs/init/muon_init044_s{}.json"), ("Muon, p = 1", "runs/knob/muon_pow1_init044_s{}.json"),
              ("Muon, p = 0.25", "runs/knob/muon_pow0.25_init044_s{}.json"),
              ("Muon, p = 0", "runs/knob/muon_pow0_init044_s{}.json")]),
}


def at_level(path, lv):
    r = load(path)
    d = json.load(open(path))
    i = np.nonzero(r.err_last <= lv)[0]
    if not len(i):
        return None
    D = d["evals"][int(i[0])]["dist_8_32"]
    pcr = {k: v for k, v in D.items() if k.startswith("pcr_")}
    rid = {k: v for k, v in D.items() if k.startswith("ridge_")}
    kp, kr = min(pcr, key=pcr.get), min(rid, key=rid.get)
    return dict(step=int(r.steps[i[0]]), ratio=pcr[kp] / rid[kr], pcr=kp, ridge=kr, d_pcr=pcr[kp], d_ridge=rid[kr],
                d_ols=D["ols"])


def main():
    out = {}
    for name, S in SETTINGS.items():
        print(f"== {name}: ratio = distance to nearest PCR / distance to nearest ridge (> 1: closer to ridge)")
        print(f"  {'arm':15s} " + " ".join(f"{'error ' + str(lv):>24s}" for lv in S["levels"]) + "   summary")
        out[name] = {}
        for arm, pat in S["arms"]:
            cells, ratios = [], []
            for lv in S["levels"]:
                vals = []
                for s in S["seeds"]:
                    p = os.path.join(ROOT, pat.format(s))
                    if not os.path.exists(p):
                        continue
                    x = at_level(p, lv)
                    if x is not None:
                        vals.append(x)
                        out[name][f"{arm}|{s}|{lv}"] = x
                ratios += [v["ratio"] for v in vals]
                cells.append(" ".join(f"{v['ratio']:.2f}" for v in vals) if vals else "-")
            if ratios:
                ridge_n = sum(r > 1 for r in ratios)
                summ = f"{min(ratios):.2f}-{max(ratios):.2f}, closer to ridge in {ridge_n}/{len(ratios)}"
            else:
                summ = "-"
            print(f"  {arm:15s} " + " ".join(f"{c:>24s}" for c in cells) + f"   {summ}")
        print()
    os.makedirs(os.path.join(ROOT, "figures/report"), exist_ok=True)
    json.dump(out, open(os.path.join(ROOT, "figures/report/algorithm.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
