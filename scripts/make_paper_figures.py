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


def err40_curves(arm):
    """Held-out error after 40 examples (Eq. 2) per evaluation step: steps, mean and s.d. over seeds."""
    rs = [load(os.path.join(ROOT, "runs/garg_scale", FILE[arm].format(s))) for s in SEEDS]
    top = 2000 if arm == "Muon" else min(int(r.steps[-1]) for r in rs)
    st = np.array([x for x in rs[0].steps if x <= top])
    E = np.array([[r.err_last[list(r.steps).index(x)] for x in st] for r in rs])
    return st, E.mean(0), E.std(0, ddof=1)


def fig_loss():
    rows = jl("report/curves.json")
    fig, axs = plt.subplots(1, 2, figsize=(W, 1.5))
    ax = axs[0]
    for arm in ORDER:
        r = [x for x in rows if x["arm"] == arm and (arm != "Muon" or x["step"] <= 2000)]
        st = np.array([x["step"] for x in r])
        for key, ls, lw in (("val", "-", 1.5), ("train", ":", 1.0)):
            m = np.array([x[key] for x in r])
            sd = np.array([x[key + "_sd"] for x in r])
            ax.plot(st, m, ls, color=COL[arm], lw=lw, label=arm if key == "val" else None)
            if key == "val":
                ax.fill_between(st, np.maximum(m - sd, 1e-3), m + sd, color=COL[arm], alpha=0.2, lw=0)
    ax.plot([], [], "k-", lw=1.5, label="validation")
    ax.plot([], [], "k:", lw=1.0, label="training")
    ax.set_yscale("log")
    ax.set_ylim(0.09, 1.1)
    ax.set_yticks([0.1, 0.2, 0.5, 1.0])
    ax.set_yticklabels(["0.1", "0.2", "0.5", "1"])
    ax.set_ylabel("loss, all 41 positions")
    ax.set_title("(a) Training and validation loss", loc="left")
    ax.legend(frameon=False, loc="upper right", ncol=2, handlelength=1.5, columnspacing=0.8)
    ax = axs[1]
    for arm in ORDER:
        st, m, sd = err40_curves(arm)
        ax.plot(st, m, color=COL[arm], lw=1.5 if arm.startswith("Muon") else 1.2)
        ax.fill_between(st, np.maximum(m - sd, 1e-3), m + sd, color=COL[arm], alpha=0.2, lw=0)
    for lv in (0.05, 0.04, 0.03):
        ax.axhline(lv, color="k", lw=0.6, ls="--")
    ax.text(3950, 0.55, "dashed: matched\nlevels 0.05, 0.04, 0.03", fontsize=6, ha="right", va="center", color="#333")
    ax.set_yscale("log")
    ax.set_ylim(0.005, 1.1)
    ax.set_yticks([0.01, 0.05, 0.1, 0.2, 0.5, 1.0])
    ax.set_yticklabels(["0.01", "0.05", "0.1", "0.2", "0.5", "1"])
    ax.minorticks_off()
    ax.set_ylabel("error after 40 examples")
    ax.set_title("(b) Held-out error used for matching", loc="left")
    for ax in axs:
        ax.set_xlim(0, 4000)
        ax.set_xlabel("training step")
        ax.grid(axis="y", lw=0.4, alpha=0.5)
    fig.tight_layout(w_pad=1.0)
    fig.savefig(os.path.join(OUT, "fig_loss.pdf"))
    plt.close(fig)


def fig_h1(level=0.04):
    P = jl("report/probe_refs.json")
    runs, refs = P["runs"], P["refs"]
    j = np.arange(1, 21)
    fig, axs = plt.subplots(1, 2, figsize=(W, 1.66), gridspec_kw=dict(width_ratios=[1.25, 1]))
    ax = axs[0]
    ax.plot(j, refs["pcr_11"], "--", color="#888", lw=0.9, drawstyle="steps-mid", label="PCR$_{11}$")
    ax.plot(j, refs["ridge_1"], "-.", color="#888", lw=0.9, label="ridge, $\\alpha=1$")
    for arm in ("AdamW", "SGD", "Muon", "Muon, p = 1"):
        F = np.array([runs[f"{arm}|{s}|{level}"] for s in SEEDS])
        m, sd = F.mean(0), F.std(0, ddof=1)
        if arm == "Muon, p = 1":
            ax.plot(j, m, color=COL[arm], lw=0.8, label=arm)
            continue
        ax.plot(j, m, color=COL[arm], lw=1.4, marker="o", ms=2, label=arm)
        ax.fill_between(j, m - sd, m + sd, color=COL[arm], alpha=0.2, lw=0)
    ax.axhline(0, color="k", lw=0.5)
    ax.set_xticks([1, 5, 10, 15, 20])
    ax.set_xlim(0.5, 20.5)
    ax.set_ylim(-1.0, 1.08)
    ax.set_xlabel("principal direction $j$ (variance $\\propto 1/j^2$)")
    ax.set_ylabel("probe score $f_j$")
    ax.set_title(f"(a) Probe profile at matched error {level}", loc="left")
    ax.legend(frameon=False, loc="lower left", ncol=1, handlelength=1.6, fontsize=6.2, labelspacing=0.25)
    ax.grid(axis="y", lw=0.4, alpha=0.5)
    ax = axs[1]
    comps = (("AdamW", "Muon $-$ AdamW"), ("SGD", "Muon $-$ SGD"), ("Muon, p = 1", "Muon $-$ ($p{=}1$)"))
    for k, (other, lab) in enumerate(comps):
        for dx, clip, mk in ((-0.13, False, "o"), (0.13, True, "s")):
            dd = []
            for s in SEEDS:
                a, b = np.array(runs[f"Muon|{s}|{level}"][2:]), np.array(runs[f"{other}|{s}|{level}"][2:])
                if clip:
                    a, b = np.clip(a, 0, None), np.clip(b, 0, None)
                dd.append(a.mean() - b.mean())
            dd = np.array(dd)
            ax.errorbar(k + dx, dd.mean(), yerr=dd.std(ddof=1), fmt=mk, ms=4, color=COL["Muon"],
                        mfc=COL["Muon"] if not clip else "white", capsize=0, lw=1.2,
                        label=("raw $\\bar f$" if not clip else "bounded $\\bar f_+$") if k == 0 else None)
            ax.scatter(np.full(5, k + dx + 0.07), dd, s=5, facecolors="none", edgecolors="#555", lw=0.5)
    ax.axhline(0, color="k", lw=0.5)
    ax.set_xticks(range(3))
    ax.set_xticklabels([c[1].replace(" $-$ ", " $-$\n") for c in comps], fontsize=6.5)
    ax.set_ylim(-0.035, 0.16)
    ax.set_xlim(-0.5, 2.5)
    ax.set_ylabel("paired difference (dirs. 3–20)")
    ax.set_title(f"(b) Score differences at error {level}", loc="left")
    ax.legend(frameon=False, loc="upper center", ncol=2, handletextpad=0.3)
    ax.grid(axis="y", lw=0.4, alpha=0.5)
    fig.tight_layout(w_pad=1.0)
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
    fig, axs = plt.subplots(1, 2, figsize=(W, 1.44))
    off = {"SGD": -0.04, "Muon": 0.0, "Muon, p = 1": 0.04}
    for arm in ("SGD", "Muon, p = 1", "Muon"):
        ss = [s for s in SEEDS if (arm, s) in Q and ("AdamW", s) in Q]
        R = np.array([(Q[arm, s] / Q["AdamW", s])[keep][order] for s in ss])
        A = np.array([(Q[arm, s] - Q["AdamW", s])[keep][order] for s in ss])
        for ax, V in ((axs[0], R), (axs[1], A)):
            m, sd = V.mean(0), V.std(0, ddof=1)
            for side in (tt < 0, tt > 0):
                ax.errorbar(tt[side] + off[arm], m[side], yerr=sd[side], fmt="o-", ms=3.2, color=COL[arm], lw=1.0,
                            capsize=0, elinewidth=1.1, label=arm if (side[-1] and ax is axs[0]) else None)
    ax = axs[0]
    ax.axhline(1, color="k", lw=0.7, ls="--")
    ax.set_yscale("log")
    ax.set_ylim(0.6, 3.0)
    ax.set_yticks([0.75, 1, 1.5, 2])
    ax.set_yticklabels(["0.75", "1", "1.5", "2"])
    ax.minorticks_off()
    ax.set_ylabel("shifted error / AdamW's")
    ax.set_title(f"(a) Ratio to AdamW at error {level}", loc="left")
    ax.legend(frameon=False, loc="upper center", ncol=3, handlelength=1.2, columnspacing=0.8, fontsize=6.3)
    ax = axs[1]
    ax.axhline(0, color="k", lw=0.7, ls="--")
    ax.set_ylabel("shifted error $-$ AdamW's")
    ax.set_title("(b) Difference from AdamW", loc="left")
    for ax in axs:
        ax.axvline(0, color="#999", lw=0.5, ls=":")
        ax.set_xticks([-1, -0.5, 0, 0.5, 1])
        ax.set_xticklabels(["$-1$", "$-0.5$", "0", "0.5", "1"])
        ax.set_xlabel("query shift $t$")
        ax.grid(axis="y", lw=0.4, alpha=0.5)
    fig.tight_layout(w_pad=1.0)
    fig.savefig(os.path.join(OUT, "fig_h2.pdf"))
    plt.close(fig)


def fig_linattn():
    d = jl("noise/linattn_matched.json")["arms"]
    band = [("gd", "SGD", "#a8a8a0", "-"), ("adam", "Adam", "#404040", "-"),
            ("muon_exact", "$p=0$ (exact)", "#0b2e6e", "-"), ("muon", "Muon (Newton–Schulz)", "#1f6fd1", "--")]
    thin = [("muon_pow1", "$p=1$", "#16a37a"), ("muon_pow0.25", "$p=0.25$", "#6aa8c8"), ("muon_pow0.1", "$p=0.1$", "#3c6fb0")]
    fig, axs = plt.subplots(1, 2, figsize=(W, 1.44), gridspec_kw=dict(width_ratios=[1.0, 0.85]))
    ax = axs[0]
    j = np.arange(1, 9)
    for key, name, c in thin:
        ax.plot(j, d[key]["profile"], color=c, lw=0.9, marker=".", ms=3, label=name)
    for key, name, c, ls in band:
        r = d[key]
        ax.plot(j, r["profile"], ls, color=c, marker="o", ms=2.5, lw=1.3, label=name)
        ax.fill_between(j, r["profile_lo"], r["profile_hi"], color=c, alpha=0.15, lw=0)
    ax.set_xlabel("principal direction $j$ (variance $\\propto 1/j^2$)")
    ax.set_ylabel("fraction of optimal gain")
    ax.set_title("(a) Gain profile at excess loss 0.1", loc="left")
    ax.set_ylim(-0.05, 1.25)
    ax.grid(axis="y", lw=0.4, alpha=0.5)
    ax.legend(frameon=False, fontsize=6, ncol=1, loc="upper left", bbox_to_anchor=(1.0, 1.02), handlelength=1.6)
    ax = axs[1]
    ps = [("muon_pow1", 1.0), ("muon_pow0.75", 0.75), ("muon_pow0.5", 0.5), ("muon_pow0.25", 0.25),
          ("muon_pow0.1", 0.1), ("muon_exact", 0.0)]
    x = np.arange(len(ps))
    med = np.array([d[k]["median"] for k, _ in ps])
    lo = np.array([d[k]["lo"] for k, _ in ps])
    hi = np.array([d[k]["hi"] for k, _ in ps])
    for key, name, c, ls in (("gd", "SGD", "#a8a8a0", ":"), ("adam", "Adam", "#404040", ":"),
                             ("muon", "Muon (NS)", "#1f6fd1", "--")):
        r = d[key]
        ax.axhline(r["median"], color=c, lw=0.8, ls=ls)
        ax.axhspan(r["lo"], r["hi"], color=c, alpha=0.12, lw=0)
        below = key == "gd"
        ax.text(-0.35, r["median"] * (0.95 if below else 1.06), name, fontsize=6, color=c if key != "gd" else "#777",
                ha="left", va="top" if below else "bottom")
    ax.errorbar(x, med, yerr=[med - lo, hi - med], fmt="o-", color="#0b2e6e", ms=3.5, capsize=0, lw=1.1)
    ax.axhline(1, color="k", lw=0.5)
    ax.set_yscale("log")
    ax.set_ylim(0.35, 6.5)
    ax.set_yticks([0.5, 1, 2, 5])
    ax.set_yticklabels(["0.5", "1", "2", "5"])
    ax.minorticks_off()
    ax.set_xticks(x)
    ax.set_xticklabels([f"{p:g}" for _, p in ps])
    ax.set_xlim(-0.4, len(ps) - 0.6)
    ax.set_xlabel("spectral power $p$ (values run; not to scale)")
    ax.set_ylabel("$R_{\\rm PCR} / R_{\\rm ridge}$")
    ax.set_title("(b) Above 1: closer to ridge", loc="left")
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
