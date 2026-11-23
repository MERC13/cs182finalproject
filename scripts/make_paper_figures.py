"""Figures for the paper (paper/figures/*.pdf), from the committed logs and the JSON written by the analysis scripts.

  python3 scripts/report_data.py            # figures/report/{curves,h1,h2}.json (Garg-scale runs, seeds 10-14)
  python3 scripts/analyze_probe_refs.py     # figures/report/probe_refs.json (probe profiles, reference estimators)
  python3 scripts/analyze_linattn_matched.py  # figures/noise/linattn_matched.json (linear attention, spectral knob)
  python3 scripts/make_paper_figures.py

fig_loss.pdf      (a) training and validation loss over all positions, (b) held-out error after 40 examples with the
                  matched levels; mean +- 1 sd over seeds (log scale)
fig_h1.pdf        (a) probe profile f_j at error 0.04 with PCR and ridge reference profiles; (b) paired Muon - other
                  differences in the raw and the bounded probe score (figures/report/probe_refs.json)
fig_h2.pdf        shifted-query error at matched error 0.04 (Eq. 2 units): (a) ratio to AdamW (log), (b) difference
fig_linattn.pdf   linear attention: gain profile at excess loss 0.1, and the PCR-vs-ridge residual ratio averaged
                  over matched excess-loss levels, across the spectral knob p (figures/noise/linattn_matched.json)
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
# One color per optimizer in every figure (categorical slots 1-3 of the validated default palette; the
# unflattened-Muon ablation is neutral gray and dashed, so it never relies on color alone).
COL = {"SGD": "#1baf7a", "AdamW": "#eb6834", "Muon": "#2a78d6", "Muon, p = 1": "#6f6e69"}
LS = {"SGD": "-", "AdamW": "-", "Muon": "-", "Muon, p = 1": "--"}
NAME = {"SGD": "SGD", "AdamW": "AdamW", "Muon": "Muon", "Muon, p = 1": "Muon, $p{=}1$"}
ORDER = ["SGD", "AdamW", "Muon, p = 1", "Muon"]
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
FILE = {"SGD": "sgd_s{}.json", "AdamW": "adamw_s{}.json", "Muon": "muon_s{}.json",
        "Muon, p = 1": "muon_pow1_lr0.0005_s{}.json"}
W = 5.5  # NeurIPS text width, inches

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


def end_label(ax, x, y, text, dy=0.0, **kw):
    """Direct label just right of a line's last point (text in ink; the line carries the color)."""
    ax.annotate(text, (x, y), xytext=(4, dy), textcoords="offset points", va="center", ha="left", fontsize=7.5,
                color=INK, annotation_clip=False, **kw)


def jl(name):
    return json.load(open(os.path.join(ROOT, "figures", name)))


def err40_curves(arm):
    """Held-out error after 40 examples (Eq. 2) per evaluation step: steps, mean and s.d. over seeds."""
    rs = [load(os.path.join(ROOT, "runs/garg_scale", FILE[arm].format(s))) for s in SEEDS]
    top = 2000 if arm == "Muon" else min(int(r.steps[-1]) for r in rs)
    st = np.array([x for x in rs[0].steps if x <= top])
    E = np.array([[r.err_last[list(r.steps).index(x)] for x in st] for r in rs])
    return st, E.mean(0), E.std(0, ddof=1)


def fig_loss():
    rows = jl("report/curves.json")
    fig, axs = plt.subplots(1, 2, figsize=(W, 2.05))
    ax = axs[0]
    for arm in ORDER:
        r = [x for x in rows if x["arm"] == arm and (arm != "Muon" or x["step"] <= 2000)]
        st = np.array([x["step"] for x in r])
        m, sd = np.array([x["val"] for x in r]), np.array([x["val_sd"] for x in r])
        ax.plot(st, m, LS[arm], color=COL[arm], lw=1.6)
        ax.fill_between(st, np.maximum(m - sd, 1e-3), m + sd, color=COL[arm], alpha=0.18, lw=0)
        ax.plot(st, [x["train"] for x in r], ":", color=COL[arm], lw=1.0)
    ax.plot([], [], "-", color=INK2, lw=1.6, label="validation")
    ax.plot([], [], ":", color=INK2, lw=1.0, label="training")
    ax.legend(frameon=False, loc="upper right")
    ax.set_yscale("log")
    ax.set_ylim(0.09, 1.1)
    ax.set_yticks([0.1, 0.2, 0.5, 1.0])
    ax.set_yticklabels(["0.1", "0.2", "0.5", "1"])
    ax.set_ylabel("loss (all positions)")
    ax.set_title("(a) Training and validation loss")
    ax = axs[1]
    ends = {}
    for arm in ORDER:
        st, m, sd = err40_curves(arm)
        ax.plot(st, m, LS[arm], color=COL[arm], lw=1.6)
        ax.fill_between(st, np.maximum(m - sd, 1e-3), m + sd, color=COL[arm], alpha=0.18, lw=0)
        ends[arm] = (st[-1], m[-1])
    for lv in (0.05, 0.04, 0.03):
        ax.axhline(lv, color=INK2, lw=0.6, ls=(0, (2, 2)))
    ax.set_yscale("log")
    ax.set_ylim(0.008, 1.1)
    ax.set_yticks([0.01, 0.03, 0.1, 0.3, 1.0])
    ax.set_yticklabels(["0.01", "0.03", "0.1", "0.3", "1"])
    ax.minorticks_off()
    ax.set_ylabel("error after 40 examples")
    ax.set_title("(b) Error used for matching")
    # direct labels: Muon at its end, the others at step 4,000 (spread to avoid overlap)
    end_label(ax, ends["Muon"][0], ends["Muon"][1], "Muon")
    lab = {"SGD": (4000, 0.052), "AdamW": (4000, 0.0315), "Muon, p = 1": (4000, 0.0235)}
    for arm, (x, y) in lab.items():
        end_label(ax, x, y, NAME[arm])
    for ax in axs:
        ax.set_xlim(0, 4000)
        ax.set_xticks([0, 1000, 2000, 3000, 4000])
        ax.set_xticklabels(["0", "1k", "2k", "3k", "4k"])
        ax.set_xlabel("training step")
        grid(ax)
    # identity for panel (a) comes from panel (b)'s labels and the shared colors; repeat as a compact legend
    handles = [plt.Line2D([], [], color=COL[a], ls=LS[a], lw=1.6) for a in ORDER]
    fig.legend(handles, [NAME[a] for a in ORDER], loc="upper center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 1.06))
    fig.tight_layout(w_pad=2.2)
    fig.savefig(os.path.join(OUT, "fig_loss.pdf"))
    plt.close(fig)


def fig_h1(level=0.04):
    P = jl("report/probe_refs.json")
    runs, refs = P["runs"], P["refs"]
    j = np.arange(1, 21)
    fig, axs = plt.subplots(1, 2, figsize=(W, 2.2), gridspec_kw=dict(width_ratios=[1.45, 1]))
    ax = axs[0]
    ax.axhspan(-1.05, 0, color="#f3f2ef", lw=0, zorder=0)
    ax.text(1.3, -0.93, "below 0: direction ignored (as PCR does)", fontsize=7, color=INK2, va="bottom")
    ax.plot(j, refs["pcr_11"], color=INK, lw=0.9, ls=(0, (4, 2)), drawstyle="steps-mid")
    ax.plot(j, refs["ridge_1"], color=INK, lw=0.9, ls=(0, (1, 1.5)))
    ax.annotate("PCR", (16.5, refs["pcr_11"][15]), xytext=(0, 5), textcoords="offset points", fontsize=7.5, ha="center")
    ax.annotate("ridge", (17, refs["ridge_1"][16]), xytext=(0, 5), textcoords="offset points", fontsize=7.5, ha="center")
    for arm in ("AdamW", "Muon, p = 1", "SGD", "Muon"):
        F = np.array([runs[f"{arm}|{s}|{level}"] for s in SEEDS])
        m, sd = F.mean(0), F.std(0, ddof=1)
        ax.plot(j, m, LS[arm], color=COL[arm], lw=1.8 if arm == "Muon" else 1.4, label=NAME[arm], zorder=3)
        if arm in ("AdamW", "Muon"):
            ax.fill_between(j, m - sd, m + sd, color=COL[arm], alpha=0.2, lw=0, zorder=2)
    ax.axhline(0, color=INK2, lw=0.6)
    ax.set_xticks([1, 5, 10, 15, 20])
    ax.set_xlim(0.5, 20.5)
    ax.set_ylim(-1.05, 1.1)
    ax.set_xlabel("input direction $j$ (strongest $\\rightarrow$ weakest)")
    ax.set_ylabel("probe score $f_j$")
    ax.set_title(f"(a) What is learned, at error {level}")
    ax.legend(frameon=False, loc="center left", bbox_to_anchor=(0.01, 0.42), fontsize=7.5, handlelength=1.8)
    grid(ax)
    ax = axs[1]
    comps = (("AdamW", "vs AdamW"), ("Muon, p = 1", "vs $p{=}1$"), ("SGD", "vs SGD"))
    for k, (other, lab) in enumerate(comps):
        for dx, clip, mk, nm in ((-0.16, False, "o", "probe score $\\bar f$"), (0.16, True, "s", "bounded score $\\bar f_+$")):
            dd = []
            for s in SEEDS:
                a, b = np.array(runs[f"Muon|{s}|{level}"][2:]), np.array(runs[f"{other}|{s}|{level}"][2:])
                if clip:
                    a, b = np.clip(a, 0, None), np.clip(b, 0, None)
                dd.append(a.mean() - b.mean())
            dd = np.array(dd)
            ax.bar(k + dx, dd.mean(), width=0.28, color=COL["Muon"] if not clip else "#9ec5f4", zorder=2,
                   label=nm if k == 0 else None)
            ax.errorbar(k + dx, dd.mean(), yerr=dd.std(ddof=1), fmt="none", ecolor=INK, elinewidth=0.9, capsize=2, zorder=3)
            ax.scatter(np.full(5, k + dx) + np.linspace(-0.07, 0.07, 5), dd, s=7, color=INK, zorder=4, lw=0)
    ax.axhline(0, color=INK2, lw=0.6)
    ax.set_xticks(range(3))
    ax.set_xticklabels([c[1] for c in comps])
    ax.set_xlim(-0.55, 2.55)
    ax.set_ylim(-0.03, 0.13)
    ax.set_ylabel("Muon's lead (directions 3--20)")
    ax.set_title("(b) Muon's lead, per seed")
    ax.legend(frameon=False, loc="upper right", fontsize=7)
    grid(ax)
    fig.tight_layout(w_pad=1.6)
    fig.savefig(os.path.join(OUT, "fig_h1.pdf"))
    plt.close(fig)


def fig_h2(level=0.04):
    sys.path.insert(0, os.path.join(ROOT, "scripts"))
    from analyze_garg_scale import q_matched
    Q = {}
    for arm in ORDER:
        for s in SEEDS:
            ts, v = q_matched(json.load(open(os.path.join(ROOT, "runs/garg_scale", FILE[arm].format(s)))), level)
            if v is not None:
                Q[arm, s] = v
    ts = np.array(ts)
    keep = ts != 0
    order = np.argsort(ts[keep])
    tt = ts[keep][order]
    fig, axs = plt.subplots(1, 2, figsize=(W, 2.1))
    off = {"SGD": -0.035, "Muon": 0.0, "Muon, p = 1": 0.035}
    for arm in ("SGD", "Muon, p = 1", "Muon"):
        ss = [s for s in SEEDS if (arm, s) in Q and ("AdamW", s) in Q]
        R = np.array([(Q[arm, s] / Q["AdamW", s])[keep][order] for s in ss])
        A = np.array([(Q[arm, s] - Q["AdamW", s])[keep][order] for s in ss])
        for ax, V in ((axs[0], R), (axs[1], A)):
            m, sd = V.mean(0), V.std(0, ddof=1)
            for side in (tt < 0, tt > 0):
                ax.errorbar(tt[side] + off[arm], m[side], yerr=sd[side], fmt="o", ls=LS[arm], ms=4, color=COL[arm],
                            lw=1.4 if arm == "Muon" else 1.1, capsize=0, elinewidth=1.0,
                            label=NAME[arm] if (side[-1] and ax is axs[0]) else None, zorder=3 if arm == "Muon" else 2)
    for ax in axs:
        ax.axvline(0, color=INK2, lw=0.6, ls=":")
        ax.set_xticks([-1, -0.5, 0, 0.5, 1])
        ax.set_xticklabels(["$-1$", "$-0.5$", "0", "0.5", "1"])
        ax.set_xlabel("query shift $t$ ($\\leftarrow$ common | rare $\\rightarrow$)")
        grid(ax)
    ax = axs[0]
    ax.axhline(1, color=INK, lw=0.8)
    ax.text(-0.2, 1.02, "AdamW", fontsize=7.5, va="bottom", ha="center")
    ax.set_yscale("log")
    ax.set_ylim(0.62, 3.2)
    ax.set_yticks([0.75, 1, 1.5, 2])
    ax.set_yticklabels(["0.75", "1", "1.5", "2"])
    ax.minorticks_off()
    ax.set_ylabel("error / AdamW's (log)")
    ax.set_title("(a) Relative to AdamW")
    ax.legend(frameon=False, loc="upper center", ncol=3, fontsize=7.5, handlelength=1.6, columnspacing=1.0)
    ax = axs[1]
    ax.axhline(0, color=INK, lw=0.8)
    ax.set_ylabel("error $-$ AdamW's")
    ax.set_title("(b) Absolute difference")
    ax.text(-0.75, 0.012, "tiny cost", fontsize=7, color=INK2, ha="center")
    ax.text(0.9, -0.044, "gain", fontsize=7, color=INK2, ha="right")
    fig.tight_layout(w_pad=1.8)
    fig.savefig(os.path.join(OUT, "fig_h2.pdf"))
    plt.close(fig)


def fig_linattn():
    d = jl("noise/linattn_matched.json")["arms"]
    arms = [("gd", "SGD"), ("adam", "AdamW"), ("muon_pow1", "Muon, p = 1"), ("muon", "Muon")]
    fig, axs = plt.subplots(1, 2, figsize=(W, 2.15), gridspec_kw=dict(width_ratios=[1.0, 1.0]))
    ax = axs[0]
    j = np.arange(1, 9)
    for key, arm in arms:
        r = d[key]
        ax.plot(j, r["profile"], LS[arm], color=COL[arm], marker="o", ms=3, lw=1.8 if arm == "Muon" else 1.3,
                label=NAME[arm].replace("AdamW", "Adam"), zorder=3)
        ax.fill_between(j, r["profile_lo"], r["profile_hi"], color=COL[arm], alpha=0.15, lw=0)
    ax.set_xlabel("input direction $j$ (strongest $\\rightarrow$ weakest)")
    ax.set_ylabel("fraction of optimal gain")
    ax.set_title("(a) What is learned")
    ax.set_ylim(-0.05, 1.25)
    ax.set_xticks(range(1, 9))
    ax.legend(frameon=False, loc="upper right", fontsize=7.5, handlelength=1.8)
    grid(ax)
    ax = axs[1]
    ps = [("muon_pow1", 1.0), ("muon_pow0.75", 0.75), ("muon_pow0.5", 0.5), ("muon_pow0.25", 0.25),
          ("muon_pow0.1", 0.1), ("muon_exact", 0.0)]
    x = np.arange(len(ps))
    med = np.array([d[k]["median"] for k, _ in ps])
    lo = np.array([d[k]["lo"] for k, _ in ps])
    hi = np.array([d[k]["hi"] for k, _ in ps])
    ax.axhspan(0.3, 1, color="#fdeee6", lw=0, zorder=0)
    ax.axhspan(1, 7, color="#e6f0fb", lw=0, zorder=0)
    ax.text(-0.35, 0.33, "PCR-like", fontsize=7.5, color=INK2, ha="left", va="bottom")
    ax.text(-0.35, 6.2, "ridge-like", fontsize=7.5, color=INK2, ha="left", va="top")
    ax.axhline(1, color=INK2, lw=0.6)
    ax.errorbar(x, med, yerr=[med - lo, hi - med], fmt="o-", color=INK, ms=4, capsize=0, lw=1.3, zorder=3)
    for key, arm in (("gd", "SGD"), ("adam", "AdamW"), ("muon", "Muon")):
        ax.axhline(d[key]["median"], color=COL[arm], lw=1.2, ls=(0, (4, 2)), zorder=2)
        ax.text(5.55, d[key]["median"], NAME[arm].replace("AdamW", "Adam"), fontsize=7.5, va="center", ha="left")
    ax.set_yscale("log")
    ax.set_ylim(0.3, 7)
    ax.set_yticks([0.5, 1, 2, 5])
    ax.set_yticklabels(["0.5", "1", "2", "5"])
    ax.minorticks_off()
    ax.set_xticks(x)
    ax.set_xticklabels([f"{p:g}" for _, p in ps])
    ax.set_xlim(-0.4, 5.4)
    ax.set_xlabel("flattening power $p$ (1: none, 0: full)")
    ax.set_ylabel("PCR fit / ridge fit (log)")
    ax.set_title("(b) Flattening switches the algorithm")
    fig.tight_layout(w_pad=2.6)
    fig.savefig(os.path.join(OUT, "fig_linattn.pdf"))
    plt.close(fig)


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    fig_loss()
    fig_h1()
    fig_h2()
    fig_linattn()
    print("wrote", sorted(os.listdir(OUT)))
