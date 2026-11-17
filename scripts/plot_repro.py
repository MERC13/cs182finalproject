"""Figures for the paper reproductions.  python scripts/plot_repro.py figures/verify"""
from __future__ import annotations

import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from plot_pilot import AXIS_NOTE, AXIS_RAMP, INK2, OPT_COLOR, logy, plt, top_legend  # noqa: E402
from matplotlib.ticker import FuncFormatter, LogLocator  # noqa: E402

OUT = sys.argv[1] if len(sys.argv) > 1 else "figures/verify"
C3 = {"gd": "#2a78d6", "signgd": "#eda100", "muon": "#1baf7a", "adam": "#eb6834", "spectral": "#1baf7a"}


def logx(ax):
    ax.set_xscale("log")
    ax.xaxis.set_major_locator(LogLocator(base=10))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))


def fig_zhang(R):
    Z = R["zhang"]
    fig, axes = plt.subplots(2, 2, figsize=(10, 6.2), sharex="col")
    for j, spec in enumerate(["1/i", "1/i2"]):
        sep, mer = Z.get(f"{spec}|sep"), Z.get(f"{spec}|merged")
        ax = axes[0][j]
        for v in (sep or mer)["L_m_pred"]:
            ax.axhline(v, color="#c9c8c2", lw=0.8)
        if sep:
            ax.plot(np.maximum(sep["t"], 1e-2), sep["loss"], color=OPT_COLOR["sgd"], label="separate rank-1 K,Q (gradient flow)")
        if mer:
            ax.plot(np.maximum(mer["t"], 1e-2), mer["loss"], color="#6250d6", label="merged KQ (gradient flow)")
        logx(ax); ax.set_ylabel("population loss / d")
        ax.set_title(f"λ∝{spec.replace('i2', 'i²')}: loss; grey lines = predicted plateaus L_m (Zhang et al.)")
        ax2 = axes[1][j]
        if sep:
            G = np.array(sep["gain"])
            for a in range(G.shape[1]):
                ax2.plot(np.maximum(sep["t"], 1e-2), G[:, a], color=AXIS_RAMP[a], lw=1.5, label=f"axis {a+1}")
        logx(ax2); ax2.set_ylim(-0.05, 1.1); ax2.set_xlabel("time (gradient-flow units)")
        ax2.set_ylabel("separate K,Q: gain / optimal gain")
    axes[0][0].legend(fontsize=7.5, loc="lower left")
    top_legend(fig, [axes[1][0]])
    fig.text(0.99, 0.005, AXIS_NOTE, ha="right", va="bottom", fontsize=7.5, color=INK2)
    fig.tight_layout(rect=(0, 0.01, 1, 0.93)); fig.savefig(f"{OUT}/repro_zhang.png", dpi=150); plt.close(fig)


def fig_ma(R):
    rows = R["ma"]
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4), sharey=True)
    for ax, rot in zip(axes, (False, True)):
        rs = [r for r in rows if r["rotated"] == rot]
        k = [r["kappa"] for r in rs]
        for m, lab in (("gd", "GD (step 1/L)"), ("signgd", "SignGD (tuned schedule)"), ("muon", "simplified Muon (tuned schedule)")):
            y = [r[m] if r[m] is not None else np.nan for r in rs]
            ax.plot(k, y, marker="o", ms=5, color=C3[m], label=lab)
        ax.set_xscale("log", base=2); logy(ax)
        ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
        ax.set_xlabel("condition number κ(S)  (effective κ = κ(S)³)")
        ax.set_title("S aligned (diagonal)" if not rot else "S rotated (Haar)")
    axes[0].set_ylabel("iterations to f/f(0) ≤ 1e-6")
    top_legend(fig, axes)
    fig.tight_layout(rect=(0, 0, 1, 0.86)); fig.savefig(f"{OUT}/repro_ma.png", dpi=150); plt.close(fig)


def fig_deeplin(R):
    runs = R["deeplin"]
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4), sharey=True)
    def worst(r):
        return max(t if t is not None else 1e9 for t in r["T90"])
    # one LR per method, chosen on the aligned panel (fastest slowest-mode), reused for the rotated panel
    best_lr = {m: min([r for r in runs if r["method"] == m and not r["rotated"]], key=worst)["lr"]
               for m in ("gd", "adam", "spectral")}
    for ax, rot in zip(axes, (False, True)):
        for m, lab in (("gd", "GD"), ("adam", "Adam"), ("spectral", "spectral GD (Muon, exact polar)")):
            rs = [r for r in runs if r["method"] == m and r["rotated"] == rot and r["lr"] == best_lr[m]]
            if not rs:
                continue
            r = rs[0]
            lab = f"{lab} (lr {r['lr']:g})"
            T = [t if t is not None else np.nan for t in r["T90"]]
            ax.plot(r["s"], T, marker="o", ms=5, color=C3[m], label=f"{lab}")
        ax.set_xscale("log", base=2); logy(ax); ax.invert_xaxis()
        ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
        ax.set_xlabel("target singular value s (small modes right)")
        ax.set_title("modes axis-aligned" if not rot else "modes rotated")
    axes[0].set_ylabel("steps until mode reaches 90%\n(same LR in both panels)")
    top_legend(fig, axes)
    fig.tight_layout(rect=(0, 0, 1, 0.86)); fig.savefig(f"{OUT}/repro_deeplin.png", dpi=150); plt.close(fig)


def fig_runs(files, labels, colors, out, title, maxstep=None, styles=None, floor=0.1168, ylo=0.1, loc="lower left"):
    fig, ax = plt.subplots(figsize=(5.4, 3.4))
    for i, (f, lab, col) in enumerate(zip(files, labels, colors)):
        if not os.path.exists(f):
            continue
        L = json.load(open(f))
        E = [e for e in L["evals"] if maxstep is None or e["step"] <= maxstep]
        ax.plot([e["step"] for e in E], [np.mean(e["test_err_k"][1:]) for e in E], color=col, label=lab,
                ls=(styles[i] if styles else "-"))
    logy(ax, ylo, 1.2)
    if floor:
        ax.axhline(floor, color=INK2, lw=0.8)
    ax.set_xlabel("training step")
    ax.set_ylabel("test error, mean over k = 1..32" + (f"\n(LS floor {floor:.3f})" if floor else ""))
    ax.set_title(title, fontsize=9.5)
    ax.legend(fontsize=7.5, loc=loc)
    fig.tight_layout(); fig.savefig(out, dpi=150); plt.close(fig)


def fig_garg(path, out, names=None):
    R = json.load(open(path))
    settings = [("standard", "in-distribution (isotropic)"), ("skewed", "skewed covariance (1/i², rotated)"),
                ("noisy", "noisy labels (σ = 1)"), ("half_subspace", "half-dimensional subspace"),
                ("scale_x2", "inputs scaled ×2"), ("orthogonal", "query orthogonal to context")]
    models = names or list(R["models"])
    pal = ["#eb6834", "#1baf7a", "#2a78d6", "#6250d6", "#e87ba4", "#008300"]
    fig, axes = plt.subplots(2, 3, figsize=(11, 6.0))
    for ax, (s, title) in zip(axes.flat, settings):
        D = R[s]
        ks = np.arange(len(D["least squares"]))
        for bname, col in (("least squares", INK2), ("3-NN", "#a8a7a1"), ("averaging", "#c9c8c2")):
            v = np.array([np.nan if x is None else x for x in D[bname]], float)
            ax.plot(ks, v, color=col, lw=1.4, ls=(0, (3, 2)) if bname != "least squares" else "-", label=bname)
        if "ridge (Bayes, λ=1)" in D:
            ax.plot(ks, D["ridge (Bayes, λ=1)"], color="#52514e", lw=1.2, ls=(0, (1, 1.5)), label="ridge (Bayes)")
        seen = set()
        for i, n in enumerate(models):
            v = np.array([np.nan if x is None else x for x in D[n]], float)
            opt = R["models"][n]["opt"]
            n_opt = sum(1 for m in models if R["models"][m]["opt"] == opt)
            if n_opt > 1:  # several seeds of one optimizer: colour by optimizer, one legend entry
                col = OPT_COLOR.get(opt, pal[i % len(pal)])
                lab = None if opt in seen else f"{opt.replace('adamw', 'AdamW').replace('muon', 'Muon')} ({n_opt} seeds)"
                seen.add(opt)
            else:
                col, lab = pal[i % len(pal)], n
            ax.plot(ks, v, color=col, lw=1.6, label=lab)
        ax.set_ylim(-0.02, 1.25); ax.set_title(title, fontsize=9.5); ax.set_xlabel("in-context examples k")
    axes[0][0].set_ylabel("relative error (1 = predict 0)"); axes[1][0].set_ylabel("relative error (1 = predict 0)")
    top_legend(fig, axes.flat)
    fig.tight_layout(rect=(0, 0, 1, 0.88)); fig.savefig(out, dpi=150); plt.close(fig)


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    rp = os.path.join(OUT, "repro_theory.json")
    if os.path.exists(rp):
        R = json.load(open(rp))
        if "zhang" in R:
            fig_zhang(R)
        if "ma" in R:
            fig_ma(R)
        if "deeplin" in R:
            fig_deeplin(R)
    for gp in glob.glob(os.path.join(OUT, "garg_eval_*.json")):
        fig_garg(gp, gp.replace(".json", ".png"))
    fig_runs(["runs/grid/adamw_iso_s0.json", "runs/repro/xie_adamw3e-4_rotparams_iso_s0.json"],
             ["AdamW (standard coordinates)", "AdamW in a randomly rotated parameter space"],
             [OPT_COLOR["adamw"], "#6250d6"], f"{OUT}/repro_xie.png",
             "Xie et al. check in our GPT-2 (isotropic inputs, lr 3e-4, same seed and data)", maxstep=10000)
    fig_runs(["runs/repro/garg_adam1e-4_cur_s0.json", "runs/repro/garg_adam1e-4_nocur_s0.json"],
             ["Adam 1e-4 with Garg curriculum (dims 2→8 by step 6k)", "Adam 1e-4, no curriculum"],
             [OPT_COLOR["adamw"], "#86b6ef"], f"{OUT}/repro_garg_curriculum.png",
             "Garg et al. training recipe (isotropic, Garg token layout)", loc="upper right")
    fig_runs(["runs/grid/sgd_iso_s0.json", "runs/repro/garg_sgd0.1_clip1_cur_s0.json", "runs/search/sgd_lr0.1_clip1_d2_s0.json",
              "runs/search/adamw_lr3e-4_d2_s0.json"],
             ["SGD, d = 8 (10k steps)", "SGD + Garg curriculum (stopped at 5k)", "SGD, d = 2", "AdamW, d = 2"],
             [OPT_COLOR["sgd"], "#86b6ef", "#104281", OPT_COLOR["adamw"]], f"{OUT}/sgd_fairness.png",
             "SGD + momentum (lr 0.1, clip 1) never leaves the plateau", styles=["-", "-", (0, (3, 2)), (0, (3, 2))],
             floor=None, ylo=0.03, loc="center right")
    print("wrote figures to", OUT)
