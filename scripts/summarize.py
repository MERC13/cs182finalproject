"""Print a compact view of one or more run logs: python scripts/summarize.py runs/x.json [...]"""
import json
import sys

import numpy as np


def best(dist, prefix):
    items = {k: v for k, v in dist.items() if k.startswith(prefix)}
    k = min(items, key=items.get)
    return k.split("_", 1)[1], items[k]


def show(path, every=1):
    L = json.load(open(path))
    c = L["config"]
    print(f"\n== {path}  [{c['opt']} lr={c['lr']} cond={c['cond']} seed={c['seed']}] status={L['status']} wall={L.get('wall')}")
    print(" step   time  train  err1-32  err@32 | probe err k=16-32 by axis (1=unused)     | ridge* (D)     pcr* (D)   gdpre*(D)  D_ols")
    for e in L["evals"][::every]:
        tk = np.array(e["test_err_k"])
        s = f"{e['step']:5d} {e['time']:6.0f} {e['train_loss'] if e['train_loss'] is not None else float('nan'):6.3f}  {tk[1:].mean():6.3f}  {tk[-1]:6.3f} |"
        if "probe_err_k" in e:
            pk = np.array(e["probe_err_k"])[:, 16:].mean(1)
            s += " " + " ".join(f"{v:4.2f}" for v in pk) + " |"
            D = e["dist_1_32"]
            for pre in ("ridge_", "pcr_", "gdpre_"):
                nm, v = best(D, pre)
                s += f" {nm:>5s}({v:.3f})"
            s += f" {D['ols']:.3f}"
        if "qshift_err" in e:
            s += "  qshift=" + ",".join(f"{v:.3f}" for v in e["qshift_err"])
        print(s)
    if "final_val_loss" in L:
        print("final val loss (mean over k):", L["final_val_loss"])


if __name__ == "__main__":
    every = 1
    args = sys.argv[1:]
    if args and args[0].startswith("--every="):
        every = int(args[0].split("=")[1])
        args = args[1:]
    for p in args:
        show(p, every)
