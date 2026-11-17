"""Analysis + figures for the linear-attention theory testbed.

  python scripts/analyze_linattn.py runs/linattn/grid.json figures/linattn
"""
from __future__ import annotations

import json
import os
import sys
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from plot_pilot import AXIS_NOTE, AXIS_RAMP, INK2, logy, top_legend  # noqa: E402  (shared style)
from matplotlib.ticker import FuncFormatter  # noqa: E402

OPTS = ["gd", "adam", "muon", "muon_exact", "signgd"]
LABEL = {"gd": "GD", "adam": "Adam", "muon": "Muon (NS5)", "muon_exact": "Muon (exact polar)", "signgd": "SignGD"}
COLOR = {"gd": "#2a78d6", "adam": "#eb6834", "muon": "#1baf7a", "muon_exact": "#1baf7a", "signgd": "#eda100"}
STYLE = {"muon_exact": (0, (4, 2))}
CONDS = ["iso", "1/i-ali", "1/i-rot", "1/i2-ali", "1/i2-rot"]
CLABEL = {"iso": "isotropic", "1/i-ali": "λ∝1/i aligned", "1/i-rot": "λ∝1/i rotated",
          "1/i2-ali": "λ∝1/i² aligned", "1/i2-rot": "λ∝1/i² rotated"}


def first(steps, mask):
    i = np.nonzero(mask)[0]
    return float(steps[i[0]]) if len(i) else np.nan


def metrics(r):
    st = np.array(r["steps"], float)
    L = np.array(r["loss"])
    rho = np.array(r["rho"]) / np.array(r["rho_star"])  # (S, d) fraction of optimal gain per axis
    lam = np.array(r["lam"])
    Lstar = 1 - float(np.sum(np.array(r["rho_star"]) * lam)) / len(lam)  # optimum loss / d
    exc = (L - Lstar) / (1 - Lstar)
    T_axis = np.array([first(st, rho[:, i] > 0.9) for i in range(rho.shape[1])])
    T_conv = first(st, exc < 0.05)
    T_half = first(st, exc < 0.5)
    # PCR vs ridge shape of the gain profile, averaged over the learning path (0.02 < excess < 0.9):
    # PCR-m = step profile (1 on the top-m axes, 0 elsewhere); ridge-a = lam / (lam + a)
    path = np.nonzero((exc > 0.02) & (exc < 0.9))[0]
    if len(path) == 0:
        path = [int(np.argmin(np.abs(exc - 0.5)))]
    steps_pr = np.array([0.0, 0.0])
    alphas = np.logspace(-3, 2, 200)
    ridge_shapes = lam[None, :] / (lam[None, :] + alphas[:, None])
    for j in path:
        prof = np.clip(rho[j], 0, 1.2)
        steps_pr[0] += min(np.sum((prof - (np.arange(len(lam)) < m)) ** 2) for m in range(len(lam) + 1))
        steps_pr[1] += np.min(np.sum((prof[None, :] - ridge_shapes) ** 2, 1))
    pcr_res, ridge_res = steps_pr / len(path)
    prof = np.clip(rho[path[len(path) // 2]], 0, 1.2)
    return dict(T_axis=T_axis, T_conv=T_conv, T_half=T_half, pcr_res=pcr_res, ridge_res=ridge_res,
                prof=prof, Lfinal=L[-1], Lstar=Lstar, qfinal=r["qloss"][-1], steps=st, exc=exc, rho=rho,
                q=np.array(r["qloss"]))


def load(path):
    R = json.load(open(path))
    sel = defaultdict(list)
    sweep = defaultdict(list)
    for r in R:
        key = (r["mode"], r["cond"], r["opt"])
        if r.get("selected"):
            sel[key].append(r)
        if r.get("sweep"):
            sweep[key].append(r)
    return sel, sweep


def table(sel):
    rows = []
    for (mode, cond, opt), rs in sorted(sel.items(), key=lambda kv: (kv[0][0], CONDS.index(kv[0][1]), OPTS.index(kv[0][2]))):
        M = [metrics(r) for r in rs if r["status"] == "done"]
        if not M:
            rows.append((mode, cond, opt, rs[0]["lr"], "diverged"))
            continue
        Ta = np.nanmedian(np.stack([m["T_axis"] for m in M]), 0)
        rows.append(dict(mode=mode, cond=cond, opt=opt, lr=rs[0]["lr"], n=len(M),
                         T_half=np.nanmedian([m["T_half"] for m in M]), T_conv=np.nanmedian([m["T_conv"] for m in M]),
                         T_first=Ta[0], T_last=Ta[-1], Lfinal=np.mean([m["Lfinal"] for m in M]),
                         Lstar=M[0]["Lstar"], qfinal=np.mean([m["qfinal"] for m in M]),
                         pcr_res=np.mean([m["pcr_res"] for m in M]), ridge_res=np.mean([m["ridge_res"] for m in M])))
    return rows


def fig_axes(sel, mode, out, conds=("iso", "1/i2-ali", "1/i2-rot"), opts=("gd", "adam", "muon", "muon_exact")):
    fig, axes = plt.subplots(len(opts), len(conds), figsize=(3.4 * len(conds), 2.3 * len(opts)), squeeze=False,
                             sharex=True, sharey=True)
    for i, o in enumerate(opts):
        for j, c in enumerate(conds):
            ax = axes[i][j]
            rs = sel.get((mode, c, o))
            if not rs or rs[0]["status"] != "done":
                ax.text(0.5, 0.5, "diverged / missing", transform=ax.transAxes, ha="center", color=INK2)
                continue
            m = metrics(rs[0])
            for a in range(m["rho"].shape[1]):
                ax.plot(np.maximum(m["steps"], 1), m["rho"][:, a], color=AXIS_RAMP[a], lw=1.5,
                        label=f"axis {a+1}")
            ax.set_xscale("log"); ax.set_ylim(-0.05, 1.15)
            if i == 0:
                ax.set_title(CLABEL[c])
            if j == 0:
                ax.set_ylabel(f"{LABEL[o]}  (lr {rs[0]['lr']:g})\ngain / optimal gain")
            if i == len(opts) - 1:
                ax.set_xlabel("training step")
    top_legend(fig, axes.flat)
    fig.legends[-1].set_bbox_to_anchor((0.5, 0.968))
    fig.text(0.99, 0.005, AXIS_NOTE + "  (λ∝1/i²: 5.24, 1.31, 0.58, 0.33, 0.21, 0.15, 0.11, 0.08)",
             ha="right", va="bottom", fontsize=7.5, color=INK2)
    fig.suptitle(f"Linear attention ({'separate rank-1 K,Q' if mode == 'sep' else 'merged KQ'}, 8 heads): "
                 "per-principal-axis gain during training", fontsize=10, fontweight="bold", y=0.995)
    fig.tight_layout(rect=(0, 0.01, 1, 0.94)); fig.savefig(out, dpi=150); plt.close(fig)


def fig_axis_times(sel, mode, out, conds=("1/i-ali", "1/i-rot", "1/i2-ali", "1/i2-rot"), opts=("gd", "adam", "muon", "muon_exact", "signgd")):
    fig, axes = plt.subplots(1, len(conds), figsize=(3.3 * len(conds), 3.0), squeeze=False, sharey=True)
    for j, c in enumerate(conds):
        ax = axes[0][j]
        for o in opts:
            rs = sel.get((mode, c, o))
            if not rs:
                continue
            M = [metrics(r) for r in rs if r["status"] == "done"]
            if not M:
                continue
            lam = np.array(rs[0]["lam"])
            T = np.stack([m["T_axis"] for m in M])
            Tm = np.nanmedian(T, 0)
            budget = rs[0]["steps"][-1]
            ax.plot(lam, Tm, marker="o", ms=4, color=COLOR[o], label=LABEL[o], ls=STYLE.get(o, "-"))
            miss = ~np.isfinite(Tm)
            if miss.any():  # axes not learned within the budget: open markers on the budget line
                ax.plot(lam[miss], np.full(miss.sum(), budget), "o", ms=5, mfc="none", mec=COLOR[o], mew=1.4)
        ax.set_xscale("log"); logy(ax); ax.invert_xaxis()
        from matplotlib.ticker import LogLocator as _LL
        ax.xaxis.set_major_locator(_LL(base=10, subs=(1.0, 3.0)))
        ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
        ax.xaxis.set_minor_formatter(FuncFormatter(lambda v, _: ""))
        ax.set_title(CLABEL[c]); ax.set_xlabel("axis variance λ (weak axes right)")
    axes[0][0].set_ylabel("steps until axis reaches 90% of optimal gain\n(open circles: not reached in budget)")
    axes[0][0].legend(fontsize=7)
    fig.tight_layout(); fig.savefig(out, dpi=150); plt.close(fig)


def q_at_matched(sel, mode, levels=(0.5, 0.25, 0.1)):
    """Query-shift loss interpolated at matched in-distribution excess loss (median over seeds)."""
    out = {}
    for (m, c, o), rs in sel.items():
        if m != mode or c == "iso":
            continue
        vals = []
        for r in rs:
            if r["status"] != "done":
                continue
            M = metrics(r)
            exc, q = M["exc"], M["q"]
            # first crossing of each level (excess decreases over training)
            row = []
            for lv in levels:
                idx = np.nonzero(exc <= lv)[0]
                if not len(idx) or idx[0] == 0:
                    row.append(np.nan); continue
                i = idx[0]
                w = (exc[i - 1] - lv) / max(exc[i - 1] - exc[i], 1e-12)
                row.append(q[i - 1] + w * (q[i] - q[i - 1]))
            vals.append(row)
        if vals:
            out[(c, o)] = np.nanmedian(np.array(vals), 0)
    return out


def fig_qtraj(sel, mode, out, conds=("1/i-ali", "1/i-rot", "1/i2-ali", "1/i2-rot"), opts=("gd", "adam", "muon", "muon_exact", "signgd")):
    fig, axes = plt.subplots(1, len(conds), figsize=(3.3 * len(conds), 3.1), squeeze=False, sharey=True)
    for ax, c in zip(axes[0], conds):
        for o in opts:
            rs = sel.get((mode, c, o))
            if not rs or rs[0]["status"] != "done":
                continue
            M = metrics(rs[0])
            ax.plot(np.maximum(M["exc"], 1e-3), M["q"], color=COLOR[o], lw=1.6, ls=STYLE.get(o, "-"), label=LABEL[o])
        ax.set_xscale("log"); ax.invert_xaxis(); ax.set_title(CLABEL[c])
        ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
        ax.set_xlabel("in-distribution excess loss (training →)")
    axes[0][0].set_ylabel("shifted-query loss (inverted spectrum)")
    axes[0][0].legend(fontsize=7)
    fig.suptitle(f"Testbed ({'separate rank-1 K,Q' if mode == 'sep' else 'merged KQ'}): query-shift loss at matched ordinary loss",
                 fontsize=10, fontweight="bold")
    fig.tight_layout(); fig.savefig(out, dpi=150); plt.close(fig)


if __name__ == "__main__":
    path, outdir = sys.argv[1], sys.argv[2]
    os.makedirs(outdir, exist_ok=True)
    sel, sweep = load(path)
    rows = table(sel)
    hdr = f"{'mode':6s} {'cond':9s} {'opt':10s} {'lr':>7s} {'n':>2s} {'T_half':>7s} {'T_conv':>7s} {'T_ax1':>7s} {'T_ax8':>7s} {'L_fin':>6s} {'L*':>6s} {'q_fin':>6s} {'PCRres':>6s} {'RIDres':>6s}"
    print(hdr)
    for r in rows:
        if isinstance(r, tuple):
            print(*r); continue
        print(f"{r['mode']:6s} {r['cond']:9s} {r['opt']:10s} {r['lr']:7.3g} {r['n']:2d} {r['T_half']:7.0f} {r['T_conv']:7.0f} "
              f"{r['T_first']:7.0f} {r['T_last']:7.0f} {r['Lfinal']:6.3f} {r['Lstar']:6.3f} {r['qfinal']:6.3f} "
              f"{r['pcr_res']:6.3f} {r['ridge_res']:6.3f}")
    json.dump([r for r in rows if isinstance(r, dict)], open(os.path.join(outdir, "table.json"), "w"), default=float)
    for mode in ("sep", "merged"):
        if any(k[0] == mode for k in sel):
            fig_axes(sel, mode, os.path.join(outdir, f"axes_{mode}.png"))
            fig_axis_times(sel, mode, os.path.join(outdir, f"axis_times_{mode}.png"))
            fig_qtraj(sel, mode, os.path.join(outdir, f"qshift_traj_{mode}.png"))
            Q = q_at_matched(sel, mode)
            print(f"\n[{mode}] shifted-query loss at matched in-distribution excess (0.5 / 0.25 / 0.1):")
            for c in CONDS[1:]:
                print(f"  {c:9s} " + "  ".join(f"{o}: " + "/".join(f"{v:.3f}" for v in Q[(c, o)]) for o in OPTS if (c, o) in Q))
    print("figures ->", outdir)
