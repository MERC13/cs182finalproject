"""Batch size (minibatch noise) vs optimizer-geometry effects, in the linear testbed and the GPT-2.

  python scripts/analyze_noise.py figures/noise

Testbed (runs/linattn/noise_sweep.json; selected LR per cell, 3 seeds), per optimizer x batch x covariance:
  T_conv     steps to excess loss < 5% (median; '>4000' if a seed never gets there)
  beta       spectral learning exponent: slope of log(time to learn axis i) vs log(1/λ_i), learned axes only
             (0 = all axes together, as Muon in theory; ~2 = GD's plateau law in this model)
  stair      time to learn the weakest axis / time to learn the strongest
  rot. cost  T_conv(rotated) / T_conv(aligned)
GPT-2 (runs/noise/*.json vs the batch-64 confirmation runs, seed 10-12): the same quantities from the
per-axis probe times of analyze_confirm.run_metrics.
"""
from __future__ import annotations

import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from analyze_linattn import metrics as la_metrics  # noqa: E402
from plot_pilot import INK2, OPT_COLOR, plt  # noqa: E402

OPTS = ["gd", "adam", "signgd", "muon"]
LAB = {"gd": "GD / SGD", "adam": "Adam", "signgd": "SignGD", "muon": "Muon"}
COL = {"gd": OPT_COLOR["sgd"], "adam": OPT_COLOR["adamw"], "signgd": "#6250d6", "muon": OPT_COLOR["muon"]}
BATCHES = [16, 64, 256, 1024]


def beta(T, lam):
    T, lam = np.asarray(T, float), np.asarray(lam, float)
    ok = np.isfinite(T) & (T > 0)
    if ok.sum() < 3:
        return np.nan
    return float(np.polyfit(np.log(1 / lam[ok]), np.log(T[ok]), 1)[0])


def med(v):
    v = np.asarray(v, float)
    return float(np.median(v)) if np.all(np.isfinite(v)) else np.inf  # inf: some seed never got there


def testbed(path):
    R = [r for r in json.load(open(path)) if r.get("selected")]
    cells = {}
    for r in R:
        m = la_metrics(r)
        k = (r["cond"], r["opt"], r["batch"])
        cells.setdefault(k, []).append(dict(lr=r["lr"], T_conv=m["T_conv"], T_axis=m["T_axis"],
                                            beta=beta(m["T_axis"], r["lam"]),
                                            stair=m["T_axis"][-1] / m["T_axis"][0],
                                            pcr=m["pcr_res"], ridge=m["ridge_res"], Lfinal=m["Lfinal"]))
    out = {}
    for (c, o, b), v in cells.items():
        out[f"{c}|{o}|{b}"] = dict(lr=v[0]["lr"], n=len(v), T_conv=med([x["T_conv"] for x in v]),
                                   beta=float(np.nanmedian([x["beta"] for x in v])),
                                   stair=med([x["stair"] for x in v]),
                                   axes_learned=[int(np.isfinite(x["T_axis"]).sum()) for x in v],
                                   pcr_minus_ridge=float(np.median([x["pcr"] - x["ridge"] for x in v])))
    return out


def muon_variants():
    """Newton-Schulz vs exact polar, momentum 0.95 vs 0, on λ ∝ 1/i² aligned (Muon is rotation-invariant)."""
    src = [("runs/linattn/noise_sweep.json", "muon", 0.95, "Newton–Schulz, momentum 0.95"),
           ("runs/linattn/noise_sweep_muonvar_m0.json", "muon", 0.0, "Newton–Schulz, no momentum"),
           ("runs/linattn/noise_sweep_muonvar_exact.json", "muon_exact", 0.95, "exact polar, momentum 0.95"),
           ("runs/linattn/noise_sweep_muonvar_m0.json", "muon_exact", 0.0, "exact polar, no momentum")]
    out = {}
    for path, opt, mom, name in src:
        if not os.path.exists(path):
            continue
        R = [r for r in json.load(open(path)) if r.get("selected") and r["opt"] == opt and r["cond"] == "1/i2-ali"
             and (r.get("muon_momentum") is None or abs(r["muon_momentum"] - mom) < 1e-9)]
        d = {}
        for b in BATCHES:
            v = [la_metrics(r) | dict(lam=r["lam"], lr=r["lr"]) for r in R if r["batch"] == b]
            if v:
                d[b] = dict(lr=v[0]["lr"], beta=float(np.nanmedian([beta(x["T_axis"], x["lam"]) for x in v])),
                            stair=med([x["T_axis"][-1] / x["T_axis"][0] for x in v]), T_conv=med([x["T_conv"] for x in v]))
        if d:
            out[name] = d
    return out


def gpt2():
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from analyze_confirm import run_metrics
    from icl.analysis import load
    rows = []
    files = sorted(glob.glob("runs/noise/*.json")) + [f for f in sorted(glob.glob("runs/confirm/*_1i2-*_s1[012].json"))
                                                      if "decay" not in f]
    for f in files:
        L = json.load(open(f))
        st = str(L.get("status", ""))
        if not (st == "done" or st.startswith("stopped")) or "cond" not in L:  # stopped: pilot ended early on purpose
            continue
        m = run_metrics(load(f))
        T = np.array(m["T_axis"], float)
        c = L["config"]
        rows.append(dict(run=os.path.basename(f), opt=c["opt"], batch=c["batch"], cond=c["cond"], seed=c["seed"],
                         T_axis=[None if not np.isfinite(t) else int(t) for t in T], beta=beta(T, L["cond"]["lam"]),
                         stair=float(T[-1] / T[0]) if np.isfinite(T[-1]) else None,
                         beta4=beta(T[:4], L["cond"]["lam"][:4]), status=st,
                         converged=m.get("T_conv"), steps=L["evals"][-1]["step"]))
    return rows


def fmt(x, nd=0):
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "never" if x is None or x == np.inf else "-"
    return f"{x:,.{nd}f}"


def main():
    out = sys.argv[1] if len(sys.argv) > 1 else "figures/noise"
    os.makedirs(out, exist_ok=True)
    res = {}
    if os.path.exists("runs/linattn/noise_sweep.json"):
        tb = testbed("runs/linattn/noise_sweep.json")
        res["testbed"] = tb
        print("== linear testbed (separate rank-1 K/Q, λ ∝ 1/i², 3 seeds at the selected LR)")
        print(f"{'opt':7s} {'batch':>5s} | {'T_conv ali':>10s} {'T_conv rot':>10s} {'rot. cost':>9s} | "
              f"{'beta ali':>8s} {'beta rot':>8s} | {'stair ali':>9s} {'stair rot':>9s} | lr ali / rot")
        for o in OPTS:
            for b in BATCHES:
                A, Rr = tb.get(f"1/i2-ali|{o}|{b}"), tb.get(f"1/i2-rot|{o}|{b}")
                if not A or not Rr:
                    continue
                cost = Rr["T_conv"] / A["T_conv"] if np.isfinite(A["T_conv"]) and np.isfinite(Rr["T_conv"]) else np.nan
                print(f"{o:7s} {b:5d} | {fmt(A['T_conv']):>10s} {fmt(Rr['T_conv']):>10s} {fmt(cost, 1):>9s} | "
                      f"{fmt(A['beta'], 2):>8s} {fmt(Rr['beta'], 2):>8s} | {fmt(A['stair'], 1):>9s} {fmt(Rr['stair'], 1):>9s} | "
                      f"{A['lr']:g} / {Rr['lr']:g}")
        fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.6))
        for o in OPTS:
            for c, ls in (("1/i2-ali", "-"), ("1/i2-rot", "--")):
                bs = [b for b in BATCHES if f"{c}|{o}|{b}" in tb]
                axes[0].plot(bs, [tb[f"{c}|{o}|{b}"]["beta"] for b in bs], ls, marker="o", ms=4, color=COL[o],
                             label=f"{LAB[o]} ({'aligned' if c.endswith('ali') else 'rotated'})")
                axes[1].plot(bs, [min(tb[f"{c}|{o}|{b}"]["T_conv"], 4500) for b in bs], ls, marker="o", ms=4,
                             color=COL[o])
        axes[0].set_ylabel("spectral learning exponent β\n(time to learn axis ∝ λ^-β)")
        axes[1].set_ylabel("steps to converge (capped at 4,500)"); axes[1].set_yscale("log")
        for a in axes:
            a.set_xscale("log", base=2); a.set_xticks(BATCHES); a.set_xticklabels(BATCHES); a.set_xlabel("batch size")
        axes[0].set_title("How sequential is learning?", fontsize=10)
        axes[1].set_title("How fast? (solid: aligned, dashed: rotated)", fontsize=10)
        h, l = axes[0].get_legend_handles_labels()
        fig.legend(h, l, loc="upper center", ncol=4, fontsize=8.5, bbox_to_anchor=(0.5, 1.0))
        axes[1].text(0.98, 0.9, "GD: not converged within 4,000 steps", transform=axes[1].transAxes, ha="right",
                     fontsize=7.5, va="top", color=INK2)
        fig.tight_layout(rect=(0, 0, 1, 0.8)); fig.savefig(os.path.join(out, "testbed_noise.png"), dpi=150); plt.close(fig)
    mv = muon_variants()
    if mv:
        res["muon_variants"] = mv
        print("\n== Muon variants in the testbed (λ ∝ 1/i² aligned; 3 seeds at the selected LR)")
        for name, d in mv.items():
            print(f"  {name:26s} " + "  ".join(f"b{b}: beta {fmt(d[b]['beta'], 2)} stair {fmt(d[b]['stair'], 1)} T_conv {fmt(d[b]['T_conv'])}"
                                              for b in BATCHES if b in d))
        fig, ax = plt.subplots(figsize=(4.8, 3.4))
        for (name, d), col, ls in zip(mv.items(), [COL["muon"], COL["muon"], "#0f6e4c", "#0f6e4c"], ["-", "--", "-", "--"]):
            bs = [b for b in BATCHES if b in d]
            ax.plot(bs, [d[b]["beta"] for b in bs], ls, marker="o", ms=4, color=col, label=name)
        ax.set_xscale("log", base=2); ax.set_xticks(BATCHES); ax.set_xticklabels(BATCHES)
        ax.set_xlabel("batch size"); ax.set_ylabel("spectral learning exponent β")
        ax.set_title("Muon: what restores sequential learning?", fontsize=10)
        ax.legend(fontsize=7.5, loc="upper right")
        fig.tight_layout(); fig.savefig(os.path.join(out, "testbed_muon_variants.png"), dpi=150); plt.close(fig)
    rows = gpt2()
    res["gpt2"] = rows
    print("\n== GPT-2 (λ ∝ 1/i², per-axis probe times)")
    for r in sorted(rows, key=lambda r: (r["opt"], r["cond"], r["batch"], r["seed"])):
        print(f"  {r['run']:34s} batch {r['batch']:4d} to step {r['steps']:5d}  beta {fmt(r['beta'], 2):>5s} (axes 1-4: {fmt(r['beta4'], 2):>5s})  stair {fmt(r['stair'], 1):>6s}  "
              f"axes {r['T_axis']}")
    json.dump(res, open(os.path.join(out, "noise.json"), "w"), indent=1, default=float)


if __name__ == "__main__":
    main()
