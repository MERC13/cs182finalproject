"""The paper's figures, from the logs in runs/. Writes figures/*.pdf.

  python scripts/figures.py
"""
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)
from icl.analysis import SEEDS, SHIFTS, at_excess, linattn, pcr_ridge_ratio, run  # noqa: E402
from icl.data import eval_prompts, spectrum  # noqa: E402
from icl.estimators import at_every_k, pcr, ridge  # noqa: E402
from icl.train import D, EVAL_SEED, N  # noqa: E402

OUT = "figures"
W = 5.5  # text width, inches
ORDER = ["SGD", "AdamW", "Muon, p = 1", "Muon"]
COL = {"SGD": "#1baf7a", "AdamW": "#eb6834", "Muon": "#2a78d6", "Muon, p = 1": "#6f6e69"}
LS = {"SGD": "-", "AdamW": "-", "Muon": "-", "Muon, p = 1": "--"}
NAME = {"SGD": "SGD", "AdamW": "AdamW", "Muon": "Muon", "Muon, p = 1": "Muon, $p{=}1$"}
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times", "Nimbus Roman", "DejaVu Serif"], "mathtext.fontset": "stix",
    "font.size": 8.5, "axes.titlesize": 9, "axes.labelsize": 8.5, "legend.fontsize": 7.5, "xtick.labelsize": 7.5,
    "ytick.labelsize": 7.5, "axes.spines.top": False, "axes.spines.right": False, "lines.linewidth": 1.6,
    "axes.edgecolor": INK2, "xtick.color": INK2, "ytick.color": INK2, "axes.labelcolor": INK, "text.color": INK,
    "axes.titlelocation": "left", "axes.titleweight": "bold", "savefig.bbox": "tight", "savefig.pad_inches": 0.03,
})


def grid(ax):
    ax.grid(axis="y", color=GRID, lw=0.6)
    ax.set_axisbelow(True)


def label(ax, x, y, text):
    ax.annotate(text, (x, y), xytext=(4, 0), textcoords="offset points", va="center", fontsize=7.5,
                annotation_clip=False)


def band(ax, x, m, sd, arm):
    ax.plot(x, m, LS[arm], color=COL[arm], lw=1.6)
    ax.fill_between(x, np.maximum(m - sd, 1e-3), m + sd, color=COL[arm], alpha=0.18, lw=0)


def curves(arm):
    """Steps reached by every seed, and per seed: training loss, validation loss over all positions, and error
    after 40 examples."""
    runs = [run(arm, s) for s in SEEDS]
    last = min(r.steps[-1] for r in runs)
    stack = lambda f: np.array([f(r)[r.steps <= last] for r in runs])
    return (runs[0].steps[runs[0].steps <= last], stack(lambda r: r.train_loss), stack(lambda r: r.err_k.mean(1)),
            stack(lambda r: r.err))


def fig_loss():
    fig, axs = plt.subplots(1, 2, figsize=(W, 2.05))
    for arm in ORDER:
        steps, train, val, err = curves(arm)
        band(axs[0], steps[1:], val[:, 1:].mean(0), val[:, 1:].std(0, ddof=1), arm)  # step 0: no training loss yet
        axs[0].plot(steps[1:], train[:, 1:].mean(0), ":", color=COL[arm], lw=1.0)
        band(axs[1], steps, err.mean(0), err.std(0, ddof=1), arm)
        if arm == "Muon":
            label(axs[1], steps[-1], err.mean(0)[-1], "Muon")
    ax = axs[0]
    ax.plot([], [], "-", color=INK2, lw=1.6, label="validation")
    ax.plot([], [], ":", color=INK2, lw=1.0, label="training")
    ax.legend(frameon=False, loc="upper right")
    ax.set_yscale("log")
    ax.set_ylim(0.09, 1.1)
    ax.set_yticks([0.1, 0.2, 0.5, 1.0], ["0.1", "0.2", "0.5", "1"])
    ax.set_ylabel("loss (all positions)")
    ax.set_title("(a) Training and validation loss")
    ax = axs[1]
    for level in (0.05, 0.04, 0.03):
        ax.axhline(level, color=INK2, lw=0.6, ls=(0, (2, 2)))
    for arm, y in (("SGD", 0.052), ("AdamW", 0.0315), ("Muon, p = 1", 0.0235)):
        label(ax, 4000, y, NAME[arm])
    ax.set_yscale("log")
    ax.set_ylim(0.008, 1.1)
    ax.set_yticks([0.01, 0.03, 0.1, 0.3, 1.0], ["0.01", "0.03", "0.1", "0.3", "1"])
    ax.minorticks_off()
    ax.set_ylabel("error after 40 examples")
    ax.set_title("(b) Error used for matching")
    for ax in axs:
        ax.set_xlim(0, 4000)
        ax.set_xticks([0, 1000, 2000, 3000, 4000], ["0", "1k", "2k", "3k", "4k"])
        ax.set_xlabel("training step")
        grid(ax)
    handles = [plt.Line2D([], [], color=COL[a], ls=LS[a], lw=1.6) for a in ORDER]
    fig.legend(handles, [NAME[a] for a in ORDER], loc="upper center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 1.06))
    fig.tight_layout(w_pad=2.2)
    fig.savefig(f"{OUT}/fig_loss.pdf")


def reference_probe(estimator):
    """Probe score f_j of a reference estimator, on the probe prompts the models are scored on."""
    lam = spectrum(D)
    probe = eval_prompts(lam, N, EVAL_SEED, SHIFTS)["probe"]
    x, y, j = probe["x"].double(), probe["y"].double(), probe["j"]
    pred = at_every_k(x, y, estimator)
    rel = torch.stack([((pred[j == i] - y[j == i]) ** 2).sum(0) / (y[j == i] ** 2).sum(0) for i in range(D)])
    return 1 - rel[:, 16:].mean(1).numpy()


def fig_h1(level=0.04):
    profile = {(arm, s): run(arm, s).at(run(arm, s).probe, level) for arm in ORDER for s in SEEDS}
    j = np.arange(1, D + 1)
    fig, axs = plt.subplots(1, 2, figsize=(W, 2.2), gridspec_kw=dict(width_ratios=[1.45, 1]))
    ax = axs[0]
    ax.axhspan(-1.05, 0, color="#f3f2ef", lw=0, zorder=0)
    ax.text(1.3, -0.93, "below 0: direction ignored (as PCR does)", fontsize=7, color=INK2, va="bottom")
    f_pcr = reference_probe(pcr(torch.tensor(spectrum(D)), 11))
    f_ridge = reference_probe(ridge(1.0))
    ax.plot(j, f_pcr, color=INK, lw=0.9, ls=(0, (4, 2)), drawstyle="steps-mid")
    ax.plot(j, f_ridge, color=INK, lw=0.9, ls=(0, (1, 1.5)))
    ax.annotate("PCR", (16.5, f_pcr[15]), xytext=(0, 5), textcoords="offset points", fontsize=7.5, ha="center")
    ax.annotate("ridge", (17, f_ridge[16]), xytext=(0, 5), textcoords="offset points", fontsize=7.5, ha="center")
    for arm in ("AdamW", "Muon, p = 1", "SGD", "Muon"):
        F = np.array([profile[arm, s] for s in SEEDS])
        m, sd = F.mean(0), F.std(0, ddof=1)
        ax.plot(j, m, LS[arm], color=COL[arm], lw=1.8 if arm == "Muon" else 1.4, label=NAME[arm], zorder=3)
        if arm in ("AdamW", "Muon"):
            ax.fill_between(j, m - sd, m + sd, color=COL[arm], alpha=0.2, lw=0, zorder=2)
    ax.axhline(0, color=INK2, lw=0.6)
    ax.set_xticks([1, 5, 10, 15, 20])
    ax.set_xlim(0.5, 20.5)
    ax.set_ylim(-1.05, 1.1)
    ax.set_xlabel("input direction $j$ (most common $\\rightarrow$ rarest)")
    ax.set_ylabel("probe score $f_j$")
    ax.set_title(f"(a) What is learned, at error {level}")
    ax.legend(frameon=False, loc="center left", bbox_to_anchor=(0.01, 0.42), handlelength=1.8)
    grid(ax)
    ax = axs[1]
    others = (("AdamW", "vs AdamW"), ("Muon, p = 1", "vs $p{=}1$"), ("SGD", "vs SGD"))
    for k, (other, _) in enumerate(others):
        for dx, bound, color, name in ((-0.16, False, COL["Muon"], "probe score $\\bar f$"),
                                       (0.16, True, "#9ec5f4", "bounded score $\\bar f_+$")):
            score = lambda f: np.mean(np.clip(f[2:], 0, None) if bound else f[2:])
            lead = np.array([score(profile["Muon", s]) - score(profile[other, s]) for s in SEEDS])
            ax.bar(k + dx, lead.mean(), width=0.28, color=color, zorder=2, label=name if k == 0 else None)
            ax.errorbar(k + dx, lead.mean(), yerr=lead.std(ddof=1), fmt="none", ecolor=INK, elinewidth=0.9, capsize=2,
                        zorder=3)
            ax.scatter(k + dx + np.linspace(-0.07, 0.07, len(SEEDS)), lead, s=7, color=INK, zorder=4, lw=0)
    ax.axhline(0, color=INK2, lw=0.6)
    ax.set_xticks(range(3), [o[1] for o in others])
    ax.set_xlim(-0.55, 2.55)
    ax.set_ylim(-0.03, 0.13)
    ax.set_ylabel("Muon's lead (directions 3--20)")
    ax.set_title("(b) Muon's lead, per seed")
    ax.legend(frameon=False, loc="upper right", fontsize=7)
    grid(ax)
    fig.tight_layout(w_pad=1.6)
    fig.savefig(f"{OUT}/fig_h1.pdf")


def fig_h2(level=0.04):
    t = np.array(SHIFTS)
    order = np.argsort(t)
    order = order[t[order] != 0]
    t = t[order]
    adamw = {s: run("AdamW", s).shift_at(level)[order] for s in SEEDS}
    fig, axs = plt.subplots(1, 2, figsize=(W, 2.1))
    offset = {"SGD": -0.035, "Muon": 0.0, "Muon, p = 1": 0.035}
    for arm in ("SGD", "Muon, p = 1", "Muon"):
        q = {s: run(arm, s).shift_at(level)[order] for s in SEEDS}
        for ax, V in ((axs[0], [q[s] / adamw[s] for s in SEEDS]), (axs[1], [q[s] - adamw[s] for s in SEEDS])):
            m, sd = np.mean(V, 0), np.std(V, 0, ddof=1)
            for side in (t < 0, t > 0):
                ax.errorbar(t[side] + offset[arm], m[side], yerr=sd[side], fmt="o", ls=LS[arm], ms=4, color=COL[arm],
                            lw=1.4 if arm == "Muon" else 1.1, elinewidth=1.0, zorder=3 if arm == "Muon" else 2,
                            label=NAME[arm] if side[-1] and ax is axs[0] else None)
    for ax in axs:
        ax.axvline(0, color=INK2, lw=0.6, ls=":")
        ax.set_xticks([-1, -0.5, 0, 0.5, 1], ["$-1$", "$-0.5$", "0", "0.5", "1"])
        ax.set_xlabel("query shift $t$ ($\\leftarrow$ common | rare $\\rightarrow$)")
        grid(ax)
    ax = axs[0]
    ax.axhline(1, color=INK, lw=0.8)
    ax.text(-0.2, 1.02, "AdamW", fontsize=7.5, va="bottom", ha="center")
    ax.set_yscale("log")
    ax.set_ylim(0.62, 3.2)
    ax.set_yticks([0.75, 1, 1.5, 2], ["0.75", "1", "1.5", "2"])
    ax.minorticks_off()
    ax.set_ylabel("error / AdamW's (log)")
    ax.set_title("(a) Relative to AdamW")
    ax.legend(frameon=False, loc="upper center", ncol=3, handlelength=1.6, columnspacing=1.0)
    ax = axs[1]
    ax.axhline(0, color=INK, lw=0.8)
    ax.set_ylabel("error $-$ AdamW's")
    ax.set_title("(b) Absolute difference")
    ax.text(-0.75, 0.012, "tiny cost", fontsize=7, color=INK2, ha="center")
    ax.text(0.9, -0.044, "gain", fontsize=7, color=INK2, ha="right")
    fig.tight_layout(w_pad=1.8)
    fig.savefig(f"{OUT}/fig_h2.pdf")


def fig_linattn():
    lam, runs = linattn()
    by_arm = lambda arm: [r for r in runs if r["arm"] == arm]
    fig, axs = plt.subplots(1, 2, figsize=(W, 2.15))
    ax = axs[0]
    j = np.arange(1, len(lam) + 1)
    for key, arm in (("sgd", "SGD"), ("adam", "AdamW"), ("muon_p1", "Muon, p = 1"), ("muon", "Muon")):
        P = np.array([at_excess(r, r["gain"], 0.1) for r in by_arm(key)])
        ax.plot(j, np.median(P, 0), LS[arm], color=COL[arm], marker="o", ms=3, lw=1.8 if arm == "Muon" else 1.3,
                label=NAME[arm].replace("AdamW", "Adam"), zorder=3)
        ax.fill_between(j, P.min(0), P.max(0), color=COL[arm], alpha=0.15, lw=0)
    ax.set_xlabel("input direction $j$ (most common $\\rightarrow$ rarest)")
    ax.set_ylabel("fraction of optimal gain")
    ax.set_title("(a) What is learned")
    ax.set_ylim(-0.05, 1.25)
    ax.set_xticks(j)
    ax.legend(frameon=False, loc="upper right", handlelength=1.8)
    grid(ax)
    ax = axs[1]
    ratio = {arm: [pcr_ridge_ratio(r, lam) for r in by_arm(arm)] for arm in {r["arm"] for r in runs}}
    powers = (1, 0.75, 0.5, 0.25, 0.1, 0)
    x = np.arange(len(powers))
    R = [ratio[f"muon_p{p:g}"] for p in powers]
    med = np.array([np.median(v) for v in R])
    ax.axhspan(0.3, 1, color="#fdeee6", lw=0, zorder=0)
    ax.axhspan(1, 7, color="#e6f0fb", lw=0, zorder=0)
    ax.text(-0.35, 0.33, "PCR-like", fontsize=7.5, color=INK2, va="bottom")
    ax.text(-0.35, 6.2, "ridge-like", fontsize=7.5, color=INK2, va="top")
    ax.axhline(1, color=INK2, lw=0.6)
    ax.errorbar(x, med, yerr=[med - [min(v) for v in R], [max(v) for v in R] - med], fmt="o-", color=INK, ms=4, lw=1.3,
                zorder=3)
    for key, arm in (("sgd", "SGD"), ("adam", "AdamW"), ("muon", "Muon")):
        m = np.median(ratio[key])
        ax.axhline(m, color=COL[arm], lw=1.2, ls=(0, (4, 2)), zorder=2)
        ax.text(5.55, m, NAME[arm].replace("AdamW", "Adam"), fontsize=7.5, va="center")
    ax.set_yscale("log")
    ax.set_ylim(0.3, 7)
    ax.set_yticks([0.5, 1, 2, 5], ["0.5", "1", "2", "5"])
    ax.minorticks_off()
    ax.set_xticks(x, [f"{p:g}" for p in powers])
    ax.set_xlim(-0.4, 5.4)
    ax.set_xlabel("flattening power $p$ (1: none, 0: full)")
    ax.set_ylabel("PCR fit / ridge fit (log)")
    ax.set_title("(b) Flattening switches the algorithm")
    fig.tight_layout(w_pad=2.6)
    fig.savefig(f"{OUT}/fig_linattn.pdf")


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    for make in (fig_loss, fig_h1, fig_h2, fig_linattn):
        make()
        plt.close("all")
    print("wrote", ", ".join(sorted(os.listdir(OUT))))
