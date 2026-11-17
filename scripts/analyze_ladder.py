"""Model ladder: where do SGD's staircase and Adam's rotation penalty survive?

  python scripts/analyze_ladder.py runs/ladder/ladder.json figures/ladder

Per arm (L = 1, 2, 4 on raw tokens; L = 2 with a learned 32-wide read-in) x optimizer x covariance, selected
LR, 2 seeds: probe-based axis times (axis j learned once its normalised probe error < 0.5), the spectral
exponent β fitted on the learned axes, the time to reach eval loss 0.25 (1 = zero predictor), and the
rotation cost T_rot / T_ali at that loss.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from icl.lintf import axis_times  # noqa: E402
from plot_pilot import INK2, OPT_COLOR, plt  # noqa: E402

ARMS = ["L1-none", "L2-none", "L4-none", "L2-learned"]
ARM_LABEL = {"L1-none": "1 layer\nraw tokens", "L2-none": "2 layers\nraw tokens", "L4-none": "4 layers\nraw tokens",
             "L2-learned": "2 layers\n+ learned read-in"}
OPTS = ["sgd", "adam", "muon"]
LAB = {"sgd": "SGD", "adam": "Adam", "muon": "Muon"}
COL = {"sgd": OPT_COLOR["sgd"], "adam": OPT_COLOR["adamw"], "muon": OPT_COLOR["muon"]}
LOSS_THR = 0.25


def beta(T, lam):
    ok = np.isfinite(T) & (T > 0)
    return float(np.polyfit(np.log(1 / np.asarray(lam)[ok]), np.log(T[ok]), 1)[0]) if ok.sum() >= 3 else np.nan


def t_loss(r, thr=LOSS_THR):
    i = np.nonzero(np.array(r["loss"]) <= thr)[0]
    return float(r["steps"][i[0]]) if len(i) else np.inf


def main():
    path, out = sys.argv[1], sys.argv[2]
    os.makedirs(out, exist_ok=True)
    R = [r for r in json.load(open(path)) if r.get("selected")]
    cell = {}
    for r in R:
        T = axis_times(r)
        cell.setdefault((r["arm"], r["opt"], r["cond"]), []).append(
            dict(lr=r["lr"], T=T, n=int(np.isfinite(T).sum()), beta=beta(T, r["lam"]), t=t_loss(r),
                 final=r["loss"][-1], status=r["status"]))
    res = {}
    print(f"{'arm':11s} {'opt':5s} | {'lr ali/rot':>13s} | {'axes learned':>12s} | {'beta ali/rot':>11s} | "
          f"{'t(loss<0.25) ali / rot':>23s} | {'rot. cost':>9s} | final loss ali/rot")
    for a in ARMS:
        for o in OPTS:
            A, B = cell.get((a, o, "1/i2-ali")), cell.get((a, o, "1/i2-rot"))
            if not A or not B:
                continue
            ta, tb = np.median([x["t"] for x in A]), np.median([x["t"] for x in B])
            cost = tb / ta if np.isfinite(ta) and np.isfinite(tb) else (np.inf if np.isfinite(ta) else np.nan)
            ba, bb = np.nanmedian([x["beta"] for x in A]), np.nanmedian([x["beta"] for x in B])
            res[f"{a}|{o}"] = dict(lr=[A[0]["lr"], B[0]["lr"]], axes=[[x["n"] for x in A], [x["n"] for x in B]],
                                   beta=[ba, bb], t=[ta, tb], cost=cost,
                                   final=[float(np.median([x["final"] for x in A])), float(np.median([x["final"] for x in B]))],
                                   T_ali=[x["T"].tolist() for x in A], T_rot=[x["T"].tolist() for x in B])
            f = lambda v: "never" if not np.isfinite(v) else f"{v:,.0f}"
            print(f"{a:11s} {o:5s} | {A[0]['lr']:>6g}/{B[0]['lr']:<6g} | {str([x['n'] for x in A]):>5s} {str([x['n'] for x in B]):>6s} | "
                  f"{ba:5.2f} {bb:5.2f} | {f(ta):>10s} / {f(tb):>10s} | {cost:9.2f} | "
                  f"{res[f'{a}|{o}']['final'][0]:.3f} / {res[f'{a}|{o}']['final'][1]:.3f}")
    json.dump(res, open(os.path.join(out, "ladder.json"), "w"), indent=1, default=float)

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.7))
    x = np.arange(len(ARMS))
    for o in OPTS:
        c = [res.get(f"{a}|{o}", {}).get("cost", np.nan) for a in ARMS]
        b = [res.get(f"{a}|{o}", {}).get("beta", [np.nan])[0] for a in ARMS]
        axes[0].plot(x, [min(v, 20) if np.isfinite(v) or v == np.inf else np.nan for v in c], "-o", ms=5,
                     color=COL[o], label=LAB[o])
        axes[1].plot(x, b, "-o", ms=5, color=COL[o], label=LAB[o])
    axes[0].axhline(1, color="#888", lw=0.8)
    axes[0].set_yscale("log"); axes[0].set_ylabel(f"rotation cost: steps to loss {LOSS_THR}\nrotated ÷ aligned (capped at 20)")
    axes[1].set_ylabel("spectral learning exponent β\n(aligned; learned axes)")
    axes[0].set_title("Does the input basis matter?", fontsize=10)
    axes[1].set_title("How sequential is learning?", fontsize=10)
    for a in axes:
        a.set_xticks(x); a.set_xticklabels([ARM_LABEL[a_] for a_ in ARMS], fontsize=8)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=3, fontsize=8.5, bbox_to_anchor=(0.5, 1.0))
    axes[1].text(0.02, 0.03, "λ ∝ 1/i², batch 256, 2,000 steps, 2 seeds", transform=axes[1].transAxes, fontsize=7.5,
                 color=INK2)
    fig.tight_layout(rect=(0, 0, 1, 0.9)); fig.savefig(os.path.join(out, "ladder.png"), dpi=150); plt.close(fig)
    print("wrote", os.path.join(out, "ladder.png"))


if __name__ == "__main__":
    main()
