"""Figures for the pilot.  python scripts/plot_pilot.py <runs_glob> <out_dir> [--sweep]"""
from __future__ import annotations

import os
import sys
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter, LogLocator

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from icl.analysis import COND_LABEL, COND_ORDER, OPT_COLOR, OPT_LABEL, OPT_ORDER, best_fit, load_all

INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e6e5e0", "#fcfcfb"
AXIS_RAMP = ["#0d366b", "#104281", "#184f95", "#1c5cab", "#256abf", "#3987e5", "#5598e7", "#86b6ef"]  # axis 1 (largest var) -> 8

plt.rcParams.update({
    "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "text.color": INK, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False, "lines.linewidth": 2.0,
    "font.size": 9, "axes.titlesize": 10, "axes.titleweight": "bold", "legend.frameon": False,
})


def column_titles(axes, titles):
    """Title each column on its first visible panel; x-labels on its last visible panel."""
    for j, t in enumerate(titles):
        vis = [i for i in range(axes.shape[0]) if axes[i][j].get_visible()]
        if vis:
            axes[vis[0]][j].set_title(t)
            axes[vis[-1]][j].set_xlabel("training step")
            axes[vis[-1]][j].tick_params(labelbottom=True)


AXIS_NOTE = "axis 1 = largest training variance … axis 8 = smallest"


def logy(ax, lo=None, hi=None):
    ax.set_yscale("log")
    if lo is not None:
        ax.set_ylim(lo, hi)
    ax.yaxis.set_major_locator(LogLocator(base=10, subs=(1.0, 3.0)))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    ax.yaxis.set_minor_formatter(FuncFormatter(lambda v, _: ""))


def top_legend(fig, axes_flat):
    h, l = [], []
    for ax in axes_flat:
        for hh, ll in zip(*ax.get_legend_handles_labels()):
            if ll not in l:
                h.append(hh); l.append(ll)
    if h:
        fig.legend(h, l, loc="upper center", ncol=len(h), bbox_to_anchor=(0.5, 1.0), fontsize=8.5)


def group(runs, key):
    g = defaultdict(list)
    for r in runs:
        g[key(r)].append(r)
    return g


def band(ax, runs, series_fn, color, label):
    """Mean line with min-max band across seeds (runs share eval steps)."""
    S = min(len(r.steps) for r in runs)
    Y = np.stack([series_fn(r)[:S] for r in runs])
    x = runs[0].steps[:S]
    ax.plot(x, Y.mean(0), color=color, label=label)
    if len(runs) > 1:
        ax.fill_between(x, Y.min(0), Y.max(0), color=color, alpha=0.18, lw=0)


def fig_sweep(runs, out):
    g = group(runs, lambda r: r.cond)
    conds = [c for c in COND_ORDER if c in g]
    fig, axes = plt.subplots(1, len(conds), figsize=(4.2 * len(conds), 3.2), squeeze=False)
    for ax, c in zip(axes[0], conds):
        go = group(g[c], lambda r: r.opt)
        for o in OPT_ORDER:
            if o not in go:
                continue
            rs = sorted(go[o], key=lambda r: r.lr)
            lr = [r.lr for r in rs]
            v = [r.final_val if r.status == "done" else np.nan for r in rs]
            ax.plot(lr, v, "-o", color=OPT_COLOR[o], label=OPT_LABEL[o], ms=5)
            for r in rs:
                if r.status != "done":
                    ax.plot([r.lr], [1.0], "x", color=OPT_COLOR[o], ms=7)
        ax.set_xscale("log"); logy(ax)
        ax.set_xlabel("learning rate"); ax.set_ylabel("final val error (mean over k)")
        ax.set_title(COND_LABEL[c])
    top_legend(fig, axes.flat)
    fig.tight_layout(rect=(0, 0, 1, 0.88)); fig.savefig(out, dpi=150); plt.close(fig)


def fig_curves(runs, out, metric="last"):
    g = group(runs, lambda r: r.cond)
    conds = [c for c in COND_ORDER if c in g]
    fig, axes = plt.subplots(1, len(conds), figsize=(3.6 * len(conds), 3.1), squeeze=False, sharey=True, sharex=True)
    for ax, c in zip(axes[0], conds):
        go = group(g[c], lambda r: r.opt)
        for o in OPT_ORDER:
            if o in go:
                fn = (lambda r: r.err_last) if metric == "last" else (lambda r: r.excess_mean())
                band(ax, go[o], fn, OPT_COLOR[o], OPT_LABEL[o])
        logy(ax, 5e-4, 1.5); ax.set_title(COND_LABEL[c]); ax.set_xlabel("training step")
    axes[0][0].set_ylabel("test error, 32 examples" if metric == "last" else "excess error over OLS (mean k=1..32)")
    top_legend(fig, axes.flat)
    fig.tight_layout(rect=(0, 0, 1, 0.92)); fig.savefig(out, dpi=150); plt.close(fig)


def fig_probes(runs, out, seed=0):
    rs = [r for r in runs if r.cfg["seed"] == seed and r.probe is not None]
    g = group(rs, lambda r: (r.opt, r.cond))
    conds = [c for c in COND_ORDER if any(k[1] == c for k in g)]
    opts = [o for o in OPT_ORDER if any(k[0] == o for k in g)]
    fig, axes = plt.subplots(len(opts), len(conds), figsize=(3.3 * len(conds), 2.6 * len(opts)),
                             squeeze=False, sharex=True, sharey=True)
    for i, o in enumerate(opts):
        for j, c in enumerate(conds):
            ax = axes[i][j]
            if (o, c) not in g:
                ax.set_visible(False); continue
            r = g[(o, c)][0]
            for a in range(r.probe.shape[1]):
                ax.plot(r.steps, r.probe_err()[:, a], color=AXIS_RAMP[a], lw=1.6, label=f"axis {a+1}")
            logy(ax, 3e-3, 2)
            if i == 0:
                ax.set_title(COND_LABEL[c])
            if j == 0:
                ax.set_ylabel(f"{OPT_LABEL[o]}\nprobe error (1 = axis unused)")
            if i == len(opts) - 1:
                ax.set_xlabel("training step")
    column_titles(axes, [COND_LABEL[c] for c in conds])
    fig.text(0.99, 0.005, AXIS_NOTE, ha='right', va='bottom', fontsize=7.5, color=INK2)
    top_legend(fig, axes.flat)
    fig.tight_layout(rect=(0, 0, 1, 0.93)); fig.savefig(out, dpi=150); plt.close(fig)


def fig_qshift(runs, out, ts=(0.0, 0.25, 0.5, 0.75, 1.0)):
    g = group(runs, lambda r: r.cond)
    conds = [c for c in COND_ORDER if c in g and c != "iso"]
    fig, axes = plt.subplots(1, len(conds), figsize=(3.4 * len(conds), 3.0), squeeze=False, sharey=True)
    for ax, c in zip(axes[0], conds):
        go = group(g[c], lambda r: r.opt)
        for o in OPT_ORDER:
            if o not in go:
                continue
            Q = np.stack([r.qshift[-1] for r in go[o]])
            ax.plot(ts, Q.mean(0), "-o", color=OPT_COLOR[o], label=OPT_LABEL[o], ms=5)
            if len(Q) > 1:
                ax.fill_between(ts, Q.min(0), Q.max(0), color=OPT_COLOR[o], alpha=0.18, lw=0)
        logy(ax); ax.set_title(COND_LABEL[c])
        ax.set_xlabel("query shift t  (0 = train spectrum, 1 = inverted)")
    axes[0][0].set_ylabel("query error, 32 examples (OLS = 0)")
    top_legend(fig, axes.flat)
    fig.tight_layout(rect=(0, 0, 1, 0.9)); fig.savefig(out, dpi=150); plt.close(fig)


def fig_algo(runs, out, seed=0):
    rs = [r for r in runs if r.cfg["seed"] == seed and r.dist is not None]
    g = group(rs, lambda r: (r.opt, r.cond))
    conds = [c for c in COND_ORDER if any(k[1] == c for k in g)]
    fig, axes = plt.subplots(2, len(conds), figsize=(3.5 * len(conds), 5.4), squeeze=False, sharex=True)
    for j, c in enumerate(conds):
        for o in OPT_ORDER:
            if (o, c) not in g:
                continue
            r = g[(o, c)][0]
            _, dr = best_fit(r, "ridge_")
            mp, dp = best_fit(r, "pcr_")
            axes[0][j].plot(r.steps, dp / np.maximum(dr, 1e-9), color=OPT_COLOR[o], label=OPT_LABEL[o])
            axes[1][j].plot(r.steps, [int(m) for m in mp], color=OPT_COLOR[o], label=OPT_LABEL[o], drawstyle="steps-post")
        logy(axes[0][j]); axes[0][j].axhline(1, color=INK2, lw=0.8)
        axes[0][j].set_title(COND_LABEL[c])
        axes[1][j].set_xlabel("training step"); axes[1][j].set_ylim(0.5, 8.5)
    axes[0][0].set_ylabel("best-PCR dist / best-ridge dist\n(<1: PCR-like, >1: ridge-like)")
    axes[1][0].set_ylabel("best-fitting PCR rank m*")
    top_legend(fig, axes.flat)
    fig.tight_layout(rect=(0, 0, 1, 0.93)); fig.savefig(out, dpi=150); plt.close(fig)


def fig_qshift_traj(runs, out):
    """H3 at matched ordinary error: shifted-query error (t=1) vs in-distribution error (t=0),
    both on the same tasks/contexts, one point per checkpoint (every qshift_every steps)."""
    g = group([r for r in runs if len(r.qshift)], lambda r: r.cond)
    conds = [c for c in COND_ORDER if c in g and c != "iso"]
    fig, axes = plt.subplots(1, len(conds), figsize=(3.5 * len(conds), 3.3), squeeze=False, sharey=True)
    for ax, c in zip(axes[0], conds):
        labelled = set()
        for r in sorted(g[c], key=lambda r: (OPT_ORDER.index(r.opt), r.cfg["seed"])):
            if r.cfg.get("muon_params") == "hidden+in":
                continue
            x, y = r.qshift[:, 0], r.qshift[:, -1]
            ax.plot(x, y, "-o", ms=3.5, lw=1.4, color=OPT_COLOR[r.opt], alpha=0.75,
                    label=None if r.opt in labelled else OPT_LABEL[r.opt] + " (one line per seed)")
            labelled.add(r.opt)
        lim = (3e-3, 1.5)
        ax.plot(lim, lim, color=INK2, lw=0.8)
        ax.set_xscale("log"); logy(ax, *lim); ax.set_xlim(*lim)
        ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
        ax.set_title(COND_LABEL[c]); ax.set_xlabel("in-distribution query error (t = 0)")
    axes[0][0].set_ylabel("shifted-query error (t = 1)\nsame tasks and contexts")
    top_legend(fig, axes.flat)
    fig.tight_layout(rect=(0, 0, 1, 0.9)); fig.savefig(out, dpi=150); plt.close(fig)


def fig_axis_times(runs, out, seed=0, thr=0.5):
    """Steps until each principal axis is learned (probe relative error < thr) vs axis variance."""
    rs = [r for r in runs if r.cfg["seed"] == seed and r.probe is not None and r.cond != "iso"]
    g = group(rs, lambda r: r.cond)
    conds = [c for c in COND_ORDER if c in g]
    fig, axes = plt.subplots(1, len(conds), figsize=(3.4 * len(conds), 3.2), squeeze=False, sharey=True)
    for ax, c in zip(axes[0], conds):
        for r in sorted(g[c], key=lambda r: OPT_ORDER.index(r.opt)):
            plus_in = r.cfg.get("muon_params") == "hidden+in"
            T = r.axis_learn_steps(thr)
            lab = OPT_LABEL[r.opt] + (" (+read-in)" if plus_in else "")
            ls = (0, (4, 2)) if plus_in else "-"
            ax.plot(r.lam, T, marker="o", ms=4, color=OPT_COLOR[r.opt], ls=ls, label=lab)
            miss = ~np.isfinite(T)
            if miss.any():
                ax.plot(r.lam[miss], np.full(miss.sum(), r.steps[-1]), "o", ms=5, mfc="none",
                        mec=OPT_COLOR[r.opt], mew=1.4)
        ax.set_xscale("log"); logy(ax); ax.invert_xaxis()
        from matplotlib.ticker import LogLocator as _LL
        ax.xaxis.set_major_locator(_LL(base=10, subs=(1.0, 3.0)))
        ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
        ax.xaxis.set_minor_formatter(FuncFormatter(lambda v, _: ""))
        ax.set_title(COND_LABEL[c]); ax.set_xlabel("axis variance λ (weak axes right)")
    axes[0][0].set_ylabel(f"steps until probe error < {thr}\n(open circles: not reached)")
    top_legend(fig, axes.flat)
    fig.tight_layout(rect=(0, 0, 1, 0.9)); fig.savefig(out, dpi=150); plt.close(fig)


def fig_iw(runs, out, seed=0):
    """Implied-weight gain per principal axis over training (rows: optimizer, cols: condition)."""
    rs = [r for r in runs if r.cfg["seed"] == seed and len(r.iw_gain)]
    g = group(rs, lambda r: (r.opt, r.cond))
    conds = [c for c in COND_ORDER if any(k[1] == c for k in g)]
    opts = [o for o in OPT_ORDER if any(k[0] == o for k in g)]
    fig, axes = plt.subplots(len(opts), len(conds), figsize=(3.3 * len(conds), 2.4 * len(opts)),
                             squeeze=False, sharex=True, sharey=True)
    for i, o in enumerate(opts):
        for j, c in enumerate(conds):
            ax = axes[i][j]
            if (o, c) not in g:
                ax.set_visible(False); continue
            r = g[(o, c)][0]
            for a in range(r.iw_gain.shape[1]):
                ax.plot(r.q_steps[-len(r.iw_gain):], r.iw_gain[:, a], color=AXIS_RAMP[a], lw=1.6,
                        label=f"axis {a+1}")
            ax.axhline(1, color=INK2, lw=0.8)
            ax.set_ylim(-0.1, 1.15)
            if i == 0:
                ax.set_title(COND_LABEL[c])
            if j == 0:
                ax.set_ylabel(f"{OPT_LABEL[o]}\nimplied-weight gain (1 = LS)")
            if i == len(opts) - 1:
                ax.set_xlabel("training step")
    column_titles(axes, [COND_LABEL[c] for c in conds])
    fig.text(0.99, 0.005, AXIS_NOTE, ha='right', va='bottom', fontsize=7.5, color=INK2)
    top_legend(fig, axes.flat)
    fig.tight_layout(rect=(0, 0, 1, 0.92)); fig.savefig(out, dpi=150); plt.close(fig)


if __name__ == "__main__":
    pattern, outdir = sys.argv[1], sys.argv[2]
    SEED = int(next((a.split("=")[1] for a in sys.argv[3:] if a.startswith("--seed=")), 0))
    os.makedirs(outdir, exist_ok=True)
    runs = load_all(pattern)
    if "--sweep" in sys.argv:
        fig_sweep(runs, os.path.join(outdir, "lr_sweep.png"))
    else:
        # standard arms only (Muon on hidden matrices); the +read-in variant is drawn only where labelled
        std = [r for r in runs if r.cfg.get("muon_params", "hidden") == "hidden"]
        fig_curves(std, os.path.join(outdir, "curves_err32.png"), "last")
        fig_curves(std, os.path.join(outdir, "curves_excess.png"), "excess")
        fig_probes(std, seed=SEED, out=os.path.join(outdir, "probes.png"))
        fig_qshift(std, os.path.join(outdir, "qshift.png"))
        fig_algo(std, seed=SEED, out=os.path.join(outdir, "algo.png"))
        fig_iw(std, seed=SEED, out=os.path.join(outdir, "implied_gain.png"))
        fig_qshift_traj(std, os.path.join(outdir, "qshift_traj.png"))
        fig_axis_times(runs, seed=SEED, out=os.path.join(outdir, "axis_times.png"))
    print("wrote figures to", outdir)
