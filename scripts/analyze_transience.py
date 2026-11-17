"""Transience of in-context learning with a finite training-task pool.

  python scripts/analyze_transience.py figures/transience 'runs/transience/*.json' 'runs/transience_ext/*.json'

Per checkpoint, averaged over context lengths k = 1..N-1 (normalised errors, noiseless targets):
  ICL score     s = (E_dMMSE - E_model) / (E_dMMSE - E_ridge)   on UNSEEN tasks
                1 = generalises like ridge, 0 = no better than the memorising dMMSE, < 0 worse.
  memorisation  m = (E_ridge - E_model) / (E_ridge - E_dMMSE)   on POOL tasks
                0 = ridge-level on the training tasks, 1 = dMMSE-level (tasks identified from memory).
                The pool error is the training loss minus the label-noise variance, so m is also a
                normalised training loss: the same m means the same fit to the training distribution.
  rel. distance r = D_ridge / (D_ridge + D_dMMSE)               on unseen tasks (0 = at ridge, 1 = at dMMSE)

Pre-registered timing (raw curves): emerge = first s >= 0.5; peak = argmax s; fade = first step after
the peak with s < 0.5; fade/emerge compares each optimizer with its own clock.
Added after seeing seed 0 (smoothed curves, 5-checkpoint moving average):
  landmarks  first step at which s >= 0.5 and m >= 0, 0.25, 0.5 -> AdamW/Muon speed-up per landmark;
  matched m  the ICL score s at the moment the run first reaches a given m (= a given training loss).
A run that resumes another (same optimizer, M, seed) replaces it when it is longer.
"""
from __future__ import annotations

import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from plot_pilot import INK2, OPT_COLOR, plt, top_legend  # noqa: E402

LABEL = {"adamw": "AdamW", "muon": "Muon", "nsgd": "normalised SGD", "sgdhid": "SGD (hidden) + AdamW"}
COLOR = dict(OPT_COLOR, nsgd="#6250d6", sgdhid="#2a78d6")
OPTS = ("adamw", "muon")
M_LEVELS = (-0.5, 0.0, 0.25, 0.5, 0.75)
LANDMARKS = (("s", 0.25, "ICL starts (s ≥ 0.25)"), ("s", 0.5, "ICL emerges (s ≥ 0.5)"), ("m", 0.0, "beats ridge on the pool (m ≥ 0)"),
             ("m", 0.25, "m ≥ 0.25"), ("m", 0.5, "memorised (m ≥ 0.5)"))


def variant(c):
    if c["opt"] == "muon" and c.get("muon_orth") == "fro":
        return "nsgd"
    if c["opt"] == "muon" and c.get("muon_orth") == "none":
        return "sgdhid"
    return c["opt"]


def smooth(v, w=5):
    """Centred moving average; the window shrinks at the ends."""
    h = w // 2
    return np.array([v[max(0, i - h): i + h + 1].mean() for i in range(len(v))])


def curves(L):
    E = [e for e in L["evals"] if "tr_fresh_err_k" in e and e["step"] > 0]
    ref = {k: float(np.mean(v[1:])) for k, v in L["tr_ref_err_k"].items()}
    st = np.array([e["step"] for e in E])
    ef = np.array([np.mean(e["tr_fresh_err_k"][1:]) for e in E])
    ep = np.array([np.mean(e["tr_pool_err_k"][1:]) for e in E])
    dr = np.array([np.mean(e["tr_fresh_dridge_k"][1:]) for e in E])
    dd = np.array([np.mean(e["tr_fresh_ddmmse_k"][1:]) for e in E])
    s = (ref["fresh_dmmse"] - ef) / (ref["fresh_dmmse"] - ref["fresh_ridge"])
    m = (ref["pool_ridge"] - ep) / (ref["pool_ridge"] - ref["pool_dmmse"])
    out = dict(step=st, s=s, m=m, r=dr / (dr + dd), ef=ef, ep=ep, ref=ref, s_sm=smooth(s), m_sm=smooth(m))
    for g in ("head", "tail"):  # Zipfian pools: memorisation of the most / least frequent tasks
        if E and f"tr_{g}_err_k" in E[0]:
            eg = np.array([np.mean(e[f"tr_{g}_err_k"][1:]) for e in E])
            out[f"m_{g}"] = (ref[f"{g}_ridge"] - eg) / (ref[f"{g}_ridge"] - ref[f"{g}_dmmse"])
    return out


def timing(c):
    """The pre-registered criterion, on the raw curves."""
    st, s = c["step"], c["s"]
    above = np.nonzero(s >= 0.5)[0]
    i_peak = int(np.argmax(s))
    after = np.nonzero((np.arange(len(s)) > i_peak) & (s < 0.5))[0] if len(above) else []
    m_half = np.nonzero(c["m"] >= 0.5)[0]
    return dict(emerge=int(st[above[0]]) if len(above) else None, peak=int(st[i_peak]), s_peak=float(s[i_peak]),
                fade=int(st[after[0]]) if len(after) else None, s_final=float(s[-1]),
                m_final=float(c["m"][-1]), memorised=int(st[m_half[0]]) if len(m_half) else None,
                r_final=float(c["r"][-1]))


def first_cross(x, v, level):
    """First x at which v reaches level, linearly interpolated between checkpoints (None if never)."""
    i = np.nonzero(v >= level)[0]
    if not len(i):
        return None
    i = int(i[0])
    if i == 0:
        return float(x[0])
    t = (level - v[i - 1]) / (v[i] - v[i - 1])
    return float(x[i - 1] + t * (x[i] - x[i - 1]))


def s_at_m(c, level):
    """Smoothed ICL score at the first moment the smoothed memorisation score reaches level."""
    m, s = c["m_sm"], c["s_sm"]
    i = np.nonzero(m >= level)[0]
    if not len(i):
        return None
    i = int(i[0])
    if i == 0:
        return float(s[0])
    t = (level - m[i - 1]) / (m[i] - m[i - 1])
    return float(s[i - 1] + t * (s[i] - s[i - 1]))


def load_runs(patterns):
    runs = {}
    for pat in patterns:
        for f in sorted(glob.glob(pat)):
            L = json.load(open(f))
            if "tr_ref_err_k" not in L or not L["evals"]:
                continue
            c = L["config"]
            key = (variant(c), c["n_tasks"], c["seed"])
            R = dict(path=f, opt=key[0], M=key[1], seed=key[2], status=L["status"], c=curves(L))
            if len(R["c"]["step"]) < 5:  # just started
                continue
            if key not in runs or R["c"]["step"][-1] > runs[key]["c"]["step"][-1]:
                runs[key] = R
    return sorted(runs.values(), key=lambda R: (R["M"], R["opt"], R["seed"]))


def stat(v):
    """'median [min-max] (n/N)' over seeds; None entries (never reached) are counted, not averaged."""
    got = [x for x in v if x is not None]
    if not got:
        return f"never (0/{len(v)})"
    rng = f" [{min(got):,.0f}–{max(got):,.0f}]" if len(got) > 1 else ""
    miss = f" ({len(got)}/{len(v)})" if len(got) < len(v) else ""
    return f"{np.median(got):,.0f}{rng}{miss}"


def main():
    out, patterns = sys.argv[1], sys.argv[2:] or ["runs/transience/*.json", "runs/transience_ext/*.json"]
    os.makedirs(out, exist_ok=True)
    runs = load_runs(patterns)
    fmt = lambda v: "-" if v is None else (f"{v:.2f}" if isinstance(v, float) else str(v))

    print("== pre-registered timing, per run (raw curves)")
    print(f"{'run':28s} {'M':>4s} {'steps':>6s} {'emerge':>7s} {'peak':>6s} {'s_peak':>6s} {'fade':>6s} "
          f"{'s_final':>7s} {'mem@':>6s} {'m_final':>7s} {'r_final':>7s} {'fade/emerge':>11s}")
    rows = []
    for R in runs:
        t = timing(R["c"])
        last = int(R["c"]["step"][-1])
        ratio = t["fade"] / t["emerge"] if t["fade"] and t["emerge"] else None
        # a run that has not faded yet has fade/emerge > last/emerge (censored)
        ratio_s = f"{ratio:.1f}" if ratio else (f">{last / t['emerge']:.1f}" if t["emerge"] else "-")
        lm = {f"{k}>={lv}": first_cross(R["c"]["step"], R["c"][k + "_sm"], lv) for k, lv, _ in LANDMARKS}
        sm = {f"s@m={lv}": s_at_m(R["c"], lv) for lv in M_LEVELS}
        rows.append(dict(run=os.path.relpath(R["path"]), opt=R["opt"], M=R["M"], seed=R["seed"], last=last,
                         status=R["status"], fade_over_emerge=ratio, fade_over_emerge_str=ratio_s,
                         **t, landmarks=lm, s_at_m=sm, ref=R["c"]["ref"]))
        name = f"{R['opt']}_M{R['M']}_s{R['seed']}" + ("" if R["status"] == "done" else " (running)")
        print(f"{name:28s} {R['M']:4d} {last:6d} {fmt(t['emerge']):>7s} {fmt(t['peak']):>6s} {fmt(t['s_peak']):>6s} "
              f"{fmt(t['fade']):>6s} {fmt(t['s_final']):>7s} {fmt(t['memorised']):>6s} {fmt(t['m_final']):>7s} "
              f"{fmt(t['r_final']):>7s} {ratio_s:>11s}")
    Ms = sorted({R["M"] for R in runs})

    print("\n== per pool size, median [range] over seeds")
    summary = {}
    for M in Ms:
        for o in OPTS:
            G = [r for r in rows if r["M"] == M and r["opt"] == o]
            if not G:
                continue
            summary[f"{o}_M{M}"] = dict(n=len(G), **{k: [r[k] for r in G] for k in (
                "emerge", "peak", "s_peak", "fade", "memorised", "s_final", "m_final", "fade_over_emerge_str", "last")})
            print(f"  M={M:<4d} {o:6s} n={len(G)}  emerge {stat([r['emerge'] for r in G])}  "
                  f"fade {stat([r['fade'] for r in G])}  mem@ {stat([r['memorised'] for r in G])}  "
                  f"s_peak {' '.join(f'{r['s_peak']:.2f}' for r in G)}  "
                  f"fade/emerge {' '.join(r['fade_over_emerge_str'] for r in G)}")

    print("\n== landmark steps (smoothed), median over seeds; speed-up = AdamW / Muon")
    speed = {}
    for M in Ms:
        for k, lv, desc in LANDMARKS:
            key = f"{k}>={lv}"
            v = {o: [r["landmarks"][key] for r in rows if r["M"] == M and r["opt"] == o] for o in OPTS}
            got = {o: [x for x in v[o] if x is not None] for o in OPTS}
            # a ratio needs the landmark in every run of both optimizers
            full = all(got[o] and len(got[o]) == len(v[o]) for o in OPTS)
            sp = float(np.median(got["adamw"]) / np.median(got["muon"])) if full else None
            speed[f"M{M} {key}"] = dict(adamw=v["adamw"], muon=v["muon"], speedup=sp)
            print(f"  M={M:<4d} {desc:32s} AdamW {stat(v['adamw']):24s} Muon {stat(v['muon']):24s} "
                  f"speed-up {fmt(sp) + '×' if sp else '-'}")

    print("\n== ICL score at matched memorisation score (= matched training loss), smoothed; mean [range]")
    matched = {}
    for M in Ms:
        for lv in M_LEVELS:
            v = {o: [r["s_at_m"][f"s@m={lv}"] for r in rows if r["M"] == M and r["opt"] == o] for o in OPTS}
            got = {o: [x for x in v[o] if x is not None] for o in OPTS}
            if not (got["adamw"] and got["muon"]):
                continue
            d = float(np.mean(got["muon"]) - np.mean(got["adamw"]))
            matched[f"M{M} m={lv}"] = dict(adamw=v["adamw"], muon=v["muon"], diff=d)
            cell = lambda g, n: f"{np.mean(g):.2f} [{min(g):.2f}–{max(g):.2f}]" + (f" ({len(g)}/{n})" if len(g) < n else "")
            print(f"  M={M:<4d} m={lv:+.2f}  AdamW {cell(got['adamw'], len(v['adamw'])):22s} "
                  f"Muon {cell(got['muon'], len(v['muon'])):22s} Muon − AdamW {d:+.2f}")

    json.dump(dict(runs=rows, summary=summary, speedup=speed, matched=matched),
              open(os.path.join(out, "transience.json"), "w"), indent=1, default=float)
    if Ms:
        figure(runs, Ms, out)


def figure(runs, Ms, out):
    fig, axes = plt.subplots(3, len(Ms), figsize=(3.2 * len(Ms), 8.4), squeeze=False)
    for j, M in enumerate(Ms):
        a_s, a_m, a_p = axes[0][j], axes[1][j], axes[2][j]
        for R in [R for R in runs if R["M"] == M and R["opt"] in COLOR]:
            col, lab = COLOR[R["opt"]], LABEL[R["opt"]]
            c = R["c"]
            a_s.plot(c["step"], c["s_sm"], color=col, lw=1.5, alpha=0.9, label=lab)
            a_m.plot(c["step"], c["m_sm"], color=col, lw=1.5, alpha=0.9, label=lab)
            keep = c["m_sm"] >= -1.6
            a_p.plot(c["m_sm"][keep], c["s_sm"][keep], color=col, lw=1.5, alpha=0.9, label=lab)
            a_p.plot(c["m_sm"][-1], c["s_sm"][-1], "o", ms=4, color=col, mec="none")
        a_s.set_title(f"M = {M} training tasks", fontsize=10)
        for a in (a_s, a_m):
            a.set_xscale("log"); a.set_xlim(100, 32000)
        a_s.axhline(1, color="#888", lw=0.8); a_s.axhline(0.5, color="#bbb", lw=0.8, ls="--")
        a_s.axhline(0, color="#888", lw=0.8, ls=":")
        a_m.axhline(1, color="#888", lw=0.8, ls=":"); a_m.axhline(0, color="#888", lw=0.8)
        a_m.axhline(0.5, color="#bbb", lw=0.8, ls="--")
        a_s.set_ylim(-0.35, 1.08); a_m.set_ylim(-1.6, 1.08)
        a_m.set_xlabel("training step (log scale)")
        # phase portrait: the two reference predictors sit at fixed points
        a_p.plot([0], [1], marker="s", ms=6, color=INK2, ls="none")
        a_p.plot([1], [0], marker="D", ms=6, color=INK2, ls="none")
        a_p.annotate("ridge", (0, 1), xytext=(-34, -3), textcoords="offset points", fontsize=8, color=INK2)
        a_p.annotate("dMMSE", (1, 0), xytext=(-34, 7), textcoords="offset points", fontsize=8, color=INK2)
        a_p.set_xlim(-1.6, 1.08); a_p.set_ylim(-0.35, 1.08)
        a_p.set_xlabel("memorisation score m\n(rises as the training loss falls)")
    axes[0][0].set_ylabel("ICL score on unseen tasks\n(1 = ridge, 0 = dMMSE)")
    axes[1][0].set_ylabel("memorisation of pool tasks\n(0 = ridge, 1 = dMMSE)")
    axes[2][0].set_ylabel("ICL score on unseen tasks\nat the same training loss")
    top_legend(fig, axes.flat)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(os.path.join(out, "transience.png"), dpi=150)
    plt.close(fig)
    print("wrote", os.path.join(out, "transience.png"))


if __name__ == "__main__":
    main()
