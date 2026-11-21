"""Figures for the paper (paper/figures/*.pdf), from the committed logs and the JSON written by the analysis scripts.

  python3 scripts/report_data.py            # figures/report/{curves,h1,h2}.json (Garg-scale runs, seeds 10-14)
  python3 scripts/analyze_schatten_pilot.py # figures/noise/schatten_pilot.json (linear attention, spectral knob)
  python3 scripts/make_paper_figures.py

fig_loss.pdf      training and validation loss, mean +- 1 sd over seeds (log scale)
fig_h1.pdf        (a) weak-direction fraction at matched error; (b) per-direction Muon - AdamW at error 0.04
fig_h2.pdf        shifted-query error / AdamW's at matched error (log scale)
fig_linattn.pdf   linear attention: per-direction fraction of the optimal gain at excess loss 0.1, and the
                  PCR-vs-ridge fit along the learning path, across the spectral knob p
"""
import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from icl.analysis import load  # noqa: E402

OUT = os.path.join(ROOT, "paper", "figures")
SEEDS = (10, 11, 12, 13, 14)
COL = {"SGD": "#a8a8a0", "AdamW": "#404040", "Muon": "#1f6fd1", "Muon, p = 1": "#16a37a"}
ORDER = ["SGD", "AdamW", "Muon, p = 1", "Muon"]
FILE = {"SGD": "sgd_s{}.json", "AdamW": "adamw_s{}.json", "Muon": "muon_s{}.json",
        "Muon, p = 1": "muon_pow1_lr0.0005_s{}.json"}
W = 5.5  # NeurIPS text width, inches

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times", "Nimbus Roman", "DejaVu Serif"], "mathtext.fontset": "stix",
    "font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8, "legend.fontsize": 7, "xtick.labelsize": 7,
    "ytick.labelsize": 7, "axes.spines.top": False, "axes.spines.right": False, "lines.linewidth": 1.3,
    "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
})


def jl(name):
    return json.load(open(os.path.join(ROOT, "figures", name)))


def fig_loss():
    rows = jl("report/curves.json")
    fig, axs = plt.subplots(1, 2, figsize=(W, 1.75), sharey=True)
    for ax, key, title in ((axs[0], "train", "Training loss"), (axs[1], "val", "Validation loss (1,000 held-out prompts)")):
        for arm in ORDER:
            r = [x for x in rows if x["arm"] == arm and (arm != "Muon" or x["step"] <= 2000)]
            st = np.array([x["step"] for x in r])
            m = np.array([x[key] for x in r])
            sd = np.array([x[key + "_sd"] for x in r])
            ax.plot(st, m, color=COL[arm], label=arm, lw=1.6 if arm.startswith("Muon") else 1.2)
            ax.fill_between(st, np.maximum(m - sd, 1e-3), m + sd, color=COL[arm], alpha=0.2, lw=0)
        ax.set_yscale("log")
        ax.set_ylim(0.09, 1.1)
        ax.set_yticks([0.1, 0.2, 0.5, 1.0])
        ax.set_yticklabels(["0.1", "0.2", "0.5", "1"])
        ax.set_xlim(0, 4000)
        ax.set_xlabel("training step")
        ax.set_title(title, loc="left")
        ax.grid(axis="y", lw=0.4, alpha=0.5)
    axs[0].set_ylabel("loss (squared error / d)")
    axs[0].legend(frameon=False, loc="upper right")
    fig.tight_layout(w_pad=1.0)
    fig.savefig(os.path.join(OUT, "fig_loss.pdf"))
    plt.close(fig)


def per_direction_diff(level=0.04):
    """Muon - AdamW fraction learned per direction (first evaluation at or below `level`), per seed."""
    D = []
    for s in SEEDS:
        prof = []
        for arm in ("Muon", "AdamW"):
            r = load(os.path.join(ROOT, "runs/garg_scale", FILE[arm].format(s)))
            i = int(np.nonzero(r.err_last <= level)[0][0])
            prof.append(1 - r.probe_err()[i])
        D.append(prof[0] - prof[1])
    return np.array(D)


def fig_h1():
    rows = jl("report/h1.json")
    levels = [0.1, 0.05, 0.04, 0.03]
    fig, axs = plt.subplots(1, 2, figsize=(W, 1.85), gridspec_kw=dict(width_ratios=[1.05, 1]))
    ax = axs[0]
    off = {"SGD": -0.27, "AdamW": -0.09, "Muon, p = 1": 0.09, "Muon": 0.27}
    for arm in ORDER:
        for k, lv in enumerate(levels):
            r = next((x for x in rows if x["arm"] == arm and x["level"] == lv and x["n"]), None)
            if r is None:
                continue
            x = k + off[arm]
            ax.errorbar(x, r["mean"], yerr=r["sd"], fmt="o", ms=4, color=COL[arm], capsize=0, lw=1.2,
                        label=arm if k == 1 else None)
            ax.scatter(np.full(len(r["seeds"]), x + 0.06), r["seeds"], s=5, facecolors="none",
                       edgecolors=COL[arm], lw=0.6)
    ax.axhline(0, color="k", lw=0.5)
    ax.set_xticks(range(len(levels)))
    ax.set_xticklabels([f"{lv}" for lv in levels])
    ax.set_xlabel("matched error (later in training $\\rightarrow$)")
    ax.set_ylabel("fraction learned (dirs. 3–20)")
    ax.set_title("(a) Weak directions learned", loc="left")
    ax.text(3 - 0.27, 0.02, "SGD:\nnot\nreached", ha="center", fontsize=5.5, color="#777")
    ax.legend(frameon=False, loc="upper left", ncol=2, handletextpad=0.2, columnspacing=0.8)
    ax.grid(axis="y", lw=0.4, alpha=0.5)
    ax = axs[1]
    D = per_direction_diff(0.04)
    m, sd = D.mean(0), D.std(0, ddof=1)
    j = np.arange(1, 21)
    ax.bar(j, m, yerr=sd, color=[COL["Muon"] if v > 0 else "#9aa" for v in m], width=0.75,
           error_kw=dict(lw=0.6, capsize=0))
    ax.axhline(0, color="k", lw=0.5)
    ax.set_xticks([1, 5, 10, 15, 20])
    ax.set_xlabel("principal direction $j$ (variance $\\propto 1/j^2$)")
    ax.set_ylabel("Muon $-$ AdamW")
    ax.set_title("(b) Per direction, matched error 0.04", loc="left")
    ax.grid(axis="y", lw=0.4, alpha=0.5)
    fig.tight_layout(w_pad=1.2)
    fig.savefig(os.path.join(OUT, "fig_h1.pdf"))
    plt.close(fig)


def fig_h2():
    rows = jl("report/h2.json")
    fig, axs = plt.subplots(1, 2, figsize=(W, 1.8), sharey=True)
    off = {"SGD": -0.04, "Muon": 0.0, "Muon, p = 1": 0.04}
    for ax, lv in zip(axs, (0.05, 0.04)):
        for arm in ("SGD", "Muon, p = 1", "Muon"):
            r = [x for x in rows if x["arm"] == arm and x["level"] == lv and x["t"] != 0 and x["mean"] is not None]
            if not r:
                continue
            n = r[0]["n"]
            for side in (lambda t: t < 0, lambda t: t > 0):
                rr = [x for x in r if side(x["t"])]
                t = np.array([x["t"] for x in rr]) + off[arm]
                ax.errorbar(t, [x["mean"] for x in rr], yerr=[x["sd"] for x in rr], fmt="o-", ms=3.5,
                            color=COL[arm], lw=1.0, capsize=0, elinewidth=1.1,
                            label=(f"{arm} ({n} seeds)" if n < 5 else arm) if side(1) else None)
        ax.axhline(1, color="k", lw=0.7, ls="--")
        ax.axvline(0, color="#999", lw=0.5, ls=":")
        ax.set_yscale("log")
        ax.set_ylim(0.5, 2.6)
        ax.set_yticks([0.5, 0.75, 1, 1.5, 2, 2.5])
        ax.set_yticklabels(["0.5", "0.75", "1", "1.5", "2", "2.5"])
        ax.minorticks_off()
        ax.set_xticks([-1, -0.5, 0, 0.5, 1])
        ax.set_xlabel("query shift $t$")
        ax.set_title(f"Matched error {lv}", loc="left")
        ax.grid(axis="y", lw=0.4, alpha=0.5)
        ax.legend(frameon=False, loc="upper right")
    axs[0].set_ylabel("shifted-query error / AdamW's")
    for ax in axs:
        ax.text(-1.05, 0.53, "toward high-variance", fontsize=6, color="#666")
        ax.text(1.05, 0.53, "toward low-variance", fontsize=6, color="#666", ha="right")
    fig.tight_layout(w_pad=1.0)
    fig.savefig(os.path.join(OUT, "fig_h2.pdf"))
    plt.close(fig)


def fig_linattn():
    d = jl("noise/schatten_pilot.json")
    arms = [("gd", "SGD", "#a8a8a0", "-"), ("adam", "Adam", "#404040", "-"), ("muon_pow1", "$p=1$", "#16a37a", "-"),
            ("muon_pow0.5", "$p=0.5$", "#5aa9c9", "-"), ("muon_pow0.25", "$p=0.25$", "#3d8fd6", "-"),
            ("muon_pow0.1", "$p=0.1$", "#2a6fc0", "-"), ("muon_exact", "$p=0$ (exact)", "#0b3d91", "-"),
            ("muon", "Muon (Newton–Schulz)", "#1f6fd1", "--")]
    fig, axs = plt.subplots(1, 2, figsize=(W, 1.8), gridspec_kw=dict(width_ratios=[1.0, 0.85]))
    ax = axs[0]
    for key, name, c, ls in arms:
        r = d[f"1/i2-ali|{key}"]
        ax.plot(np.arange(1, 9), r["profile"], ls, color=c, marker="o", ms=2.5, lw=1.1, label=name)
    ax.set_xlabel("principal direction $j$ (variance $\\propto 1/j^2$)")
    ax.set_ylabel("fraction of optimal gain")
    ax.set_title("(a) Per-direction profile at excess loss 0.1", loc="left")
    ax.set_ylim(-0.05, 1.15)
    ax.grid(axis="y", lw=0.4, alpha=0.5)
    ax.legend(frameon=False, fontsize=6, ncol=1, loc="upper left", bbox_to_anchor=(1.0, 1.02), handlelength=1.6)
    ax = axs[1]
    ps = [("muon_pow1", 1.0), ("muon_pow0.75", 0.75), ("muon_pow0.5", 0.5), ("muon_pow0.25", 0.25),
          ("muon_pow0.1", 0.1), ("muon_exact", 0.0)]
    x = np.arange(len(ps))
    ax.plot(x, [d[f"1/i2-ali|{k}"]["ridge_fit"] for k, _ in ps], "o-", color="#1f6fd1", ms=3.5)
    for key, name, c in (("gd", "SGD", "#a8a8a0"), ("adam", "Adam", "#404040"), ("muon", "Muon (NS)", "#1f6fd1")):
        v = d[f"1/i2-ali|{key}"]["ridge_fit"]
        ax.axhline(v, color=c, lw=0.8, ls="--" if key == "muon" else ":")
        ax.text(-0.1, v * 1.15, name, fontsize=6, color=c, ha="left")
    ax.axhline(1, color="k", lw=0.5)
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{p:g}" for _, p in ps])
    ax.set_xlabel("spectral power $p$ (1 = no flattening)")
    ax.set_ylabel("PCR residual / ridge residual")
    ax.set_title("(b) Closer to ridge above 1", loc="left")
    ax.grid(axis="y", lw=0.4, alpha=0.5)
    fig.tight_layout(w_pad=0.5)
    fig.savefig(os.path.join(OUT, "fig_linattn.pdf"))
    plt.close(fig)


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    fig_loss()
    fig_h1()
    fig_h2()
    fig_linattn()
    print("wrote", sorted(os.listdir(OUT)))
