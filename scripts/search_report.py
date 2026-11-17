"""Rank hyperparameter configurations per (condition, optimizer).

AUC = time-average of the test error (mean over k = 1..32) over steps [0, B], by trapezoid
integration (robust to different evaluation spacings).  Also: first step below 0.9 (escape),
error at B, and the step at which the weakest principal axis' probe error falls below 0.5.

  python scripts/search_report.py [--budget adamw=4000,muon=3000,sgd=4000]
"""
from __future__ import annotations

import glob
import json
import sys
from collections import defaultdict

import numpy as np

PATTERNS = ["runs/search/*.json", "runs/sweep/*.json", "runs/plateau/*sep*.json", "runs/grid/*_s0.json"]
BUDGET = {"adamw": 4000, "muon": 3000, "sgd": 4000}


def key(c):
    k = [f"lr={c['lr']:g}"]
    if c["opt"] in ("adamw", "muon") and abs(c.get("beta2", 0.999) - 0.999) > 1e-9:
        k.append(f"b2={c['beta2']:g}")
    if c["opt"] == "muon":
        if abs(c.get("aux_lr", 1e-3) - 1e-3) > 1e-12:
            k.append(f"aux={c['aux_lr']:g}")
        if c.get("muon_params", "hidden") != "hidden":
            k.append("+readin")
        if abs(c.get("muon_momentum", 0.95) - 0.95) > 1e-9:
            k.append(f"mom={c['muon_momentum']:g}")
    if c["opt"] == "sgd":
        k.append(f"mom={c.get('momentum', 0.9):g}")
    if c.get("clip"):
        k.append(f"clip={c['clip']:g}")
    if c.get("warmup", 100) != 100:
        k.append(f"warmup={c['warmup']}")
    if not c.get("bf16", 1):
        k.append("fp32")
    if c.get("param_rotation"):
        k.append("ROTATED-PARAMS")
    if c.get("curriculum", "none") != "none":
        k.append(f"cur={c['curriculum']}")
    return " ".join(k)


def analyze(L, B):
    E = [e for e in L["evals"] if e["step"] <= B]
    st = np.array([e["step"] for e in E], float)
    err = np.array([np.mean(e["test_err_k"][1:]) for e in E])
    if st[-1] < B:  # run stopped early (pruned/diverged/killed): hold its last value (pessimistic for plateaus)
        st = np.append(st, B)
        err = np.append(err, err[-1] if not L["status"].startswith("diverged") else 1.0)
    auc = np.trapezoid(err, st) / B
    esc = next((s for s, v in zip(st, err) if v < 0.9), None)
    t_weak = None
    P = [(e["step"], np.mean(np.array(e["probe_err_k"])[-1, 16:])) for e in E if "probe_err_k" in e]
    if P:
        t_weak = next((s for s, v in P if v < 0.5), None)
    return auc, esc, err[-1], t_weak


def main():
    budget = dict(BUDGET)
    for arg in sys.argv[1:]:
        if arg.startswith("--budget="):
            budget.update({k: int(v) for k, v in (kv.split("=") for kv in arg.split("=", 1)[1].split(","))})
    rows = defaultdict(dict)
    for pat in PATTERNS:
        for f in sorted(glob.glob(pat)):
            try:
                L = json.load(open(f))
            except Exception:
                continue
            c = L["config"]
            if c.get("y_mode") != "sep" or c.get("seed", 0) != 0 or c.get("d", 8) != 8:
                continue
            B = budget[c["opt"]]
            if L["evals"][-1]["step"] < B and L["status"] == "running":
                continue  # still training
            k = key(c)
            res = analyze(L, B)
            prev = rows[(c["cond"], c["opt"])].get(k)
            if prev is None or L["evals"][-1]["step"] > prev[1]:
                rows[(c["cond"], c["opt"])][k] = (res, L["evals"][-1]["step"], L["status"], f)
    for (cond, opt), d in sorted(rows.items()):
        print(f"\n== {cond} / {opt}  (budget {budget[opt]} steps; ranked by AUC)")
        print(f"   {'config':38s} {'AUC':>6s} {'escape':>7s} {'err@B':>7s} {'weak-axis':>9s}  status")
        for k, ((auc, esc, eB, tw), last, status, f) in sorted(d.items(), key=lambda kv: kv[1][0][0]):
            print(f"   {k:38s} {auc:6.3f} {str(esc):>7s} {eB:7.3f} {str(tw):>9s}  {status}")


if __name__ == "__main__":
    main()
