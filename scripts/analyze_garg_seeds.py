"""Do H1-H3 hold across seeds at Garg et al.'s scale? Seeds 10-12 (jobs/garg_scale.txt, garg_scale_knob.txt,
garg_scale_seeds.txt).

  python scripts/analyze_garg_seeds.py

At each matched in-distribution error level (error with 40 examples, relative to predicting 0), per seed:
  H1  weak-direction fraction learned: 1 - probe error, mean over directions 3-20, at the first evaluation at or
      below the level. Holds in a seed if Muon > AdamW and Muon > SGD.
  H2  shifted-query error relative to AdamW of the same seed, at the first crossing of the level on the
      shifted-query prompts (log-linear interpolation); geometric mean over the rare-ward shifts (t > 0) and over
      the common-ward ones (t < 0). Holds in a seed if Muon's rare-ward ratio < 1 and its common-ward ratio > 1.
  H3  H1's statistic for Muon minus Muon with unflattened updates (p = 1). Holds in a seed if > 0.
Paired differences per seed are printed with one-sided paired t-tests (n - 1 = 2 degrees of freedom).
"""
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
from icl.analysis import load  # noqa: E402
from analyze_garg_scale import q_matched  # noqa: E402

DIR = "runs/garg_scale"
SEEDS = (10, 11, 12)
ARMS = [("SGD", "sgd_s{}.json"), ("AdamW", "adamw_s{}.json"), ("Muon", "muon_s{}.json"),
        ("Muon, p = 1", "muon_pow1_lr0.0005_s{}.json")]
LEVELS = (0.1, 0.05, 0.04, 0.03)


def fmt(v, w=6, p=3):
    return f"{'-':>{w}s}" if v is None or not np.isfinite(v) else f"{v:{w}.{p}f}"


def p_one_sided(diffs):
    """One-sided paired t-test that the mean difference is > 0 (Student t, n - 1 degrees of freedom)."""
    d = np.asarray(diffs, dtype=float)
    n = len(d)
    if n < 2 or d.std(ddof=1) == 0:
        return float("nan")
    t = d.mean() / (d.std(ddof=1) / np.sqrt(n))
    k = n - 1
    if k == 2:  # closed form
        return float(0.5 * (1 - t / np.sqrt(t * t + 2)))
    from math import gamma, pi
    c = gamma((k + 1) / 2) / (np.sqrt(k * pi) * gamma(k / 2))
    x = np.linspace(t, t + 200, 400001)
    return float(np.trapezoid(c * (1 + x * x / k) ** (-(k + 1) / 2), x))


def summary(vals, p=3):
    v = np.array([x for x in vals if x is not None and np.isfinite(x)])
    if len(v) == 0:
        return "-"
    return f"{v.mean():.{p}f} ± {v.std(ddof=1):.{p}f}" if len(v) > 1 else f"{v[0]:.{p}f}"


def main():
    runs = {}
    for name, pat in ARMS:
        for s in SEEDS:
            p = os.path.join(ROOT, DIR, pat.format(s))
            if os.path.exists(p):
                runs[name, s] = (load(p), json.load(open(p)))
    print("status (last step, error)")
    for name, _ in ARMS:
        cells = []
        for s in SEEDS:
            if (name, s) in runs:
                r, d = runs[name, s]
                cells.append(f"s{s}: {d.get('status', '?'):7s} {int(r.steps[-1]):5d} {float(r.err_last[-1]):.3f}")
            else:
                cells.append(f"s{s}: {'missing':21s}")
        print(f"  {name:12s} " + "   ".join(cells))

    print("\nfirst step at error 0.05, per seed")
    for name, _ in ARMS:
        st = []
        for s in SEEDS:
            if (name, s) in runs:
                r = runs[name, s][0]
                i = np.nonzero(r.err_last <= 0.05)[0]
                st.append(int(r.steps[i[0]]) if len(i) else None)
            else:
                st.append(None)
        print(f"  {name:12s} " + " ".join(f"{'-' if v is None else v:>6}" for v in st))

    tally = {}
    for lv in LEVELS:
        print(f"\n== matched in-distribution error {lv}")
        weak, rare, common = {}, {}, {}
        for (name, s), (r, d) in runs.items():
            i = np.nonzero(r.err_last <= lv)[0]
            weak[name, s] = float(np.mean(1 - r.probe_err()[int(i[0])][2:])) if len(i) else None
        for s in SEEDS:
            if ("AdamW", s) not in runs:
                continue
            ts, base = q_matched(runs["AdamW", s][1], lv)
            if base is None:
                continue
            ts = np.array(ts)
            for name, _ in ARMS:
                if (name, s) not in runs:
                    continue
                _, q = q_matched(runs[name, s][1], lv)
                if q is None:
                    continue
                ratio = q / base
                rare[name, s] = float(np.exp(np.mean(np.log(ratio[ts > 0]))))
                common[name, s] = float(np.exp(np.mean(np.log(ratio[ts < 0]))))
        print(f"  {'':12s} {'H1: weak-direction fraction learned':>40s}   {'H2: rare-ward / common-ward error vs AdamW':>48s}")
        print(f"  {'':12s} " + " ".join(f"{'s' + str(s):>6s}" for s in SEEDS) + f" {'mean ± sd':>17s}   "
              + " ".join(f"{'s' + str(s):>13s}" for s in SEEDS) + f" {'geo-mean':>13s}")
        for name, _ in ARMS:
            w = [weak.get((name, s)) for s in SEEDS]
            rc = [(rare.get((name, s)), common.get((name, s))) for s in SEEDS]
            gm = lambda xs: (float(np.exp(np.mean(np.log(xs)))) if xs else None)
            rr = gm([a for a, _ in rc if a is not None])
            cc = gm([b for _, b in rc if b is not None])
            print(f"  {name:12s} " + " ".join(fmt(v) for v in w) + f" {summary(w):>17s}   "
                  + " ".join(f"{fmt(a, 5, 2)} / {fmt(b, 5, 2)}" for a, b in rc) + f" {fmt(rr, 5, 2)} / {fmt(cc, 5, 2)}")
        h1 = [s for s in SEEDS if all(weak.get(k) is not None for k in (("Muon", s), ("AdamW", s), ("SGD", s)))]
        h1_ok = [s for s in h1 if weak["Muon", s] > weak["AdamW", s] and weak["Muon", s] > weak["SGD", s]]
        h1b = [s for s in SEEDS if weak.get(("Muon", s)) is not None and weak.get(("AdamW", s)) is not None]
        h1b_ok = [s for s in h1b if weak["Muon", s] > weak["AdamW", s]]
        h2 = [s for s in SEEDS if ("Muon", s) in rare]
        h2_ok = [s for s in h2 if rare["Muon", s] < 1 and common["Muon", s] > 1]
        h3 = [s for s in SEEDS if weak.get(("Muon", s)) is not None and weak.get(("Muon, p = 1", s)) is not None]
        d3 = [weak["Muon", s] - weak["Muon, p = 1", s] for s in h3]
        h3_ok = [x for x in d3 if x > 0]
        print(f"  H1 Muon > AdamW and SGD in {len(h1_ok)}/{len(h1)} seeds (Muon > AdamW in {len(h1b_ok)}/{len(h1b)});"
              f"  H2 rare < 1 < common in {len(h2_ok)}/{len(h2)};"
              f"  H3 Muon - p=1 = {summary(d3)} (> 0 in {len(h3_ok)}/{len(h3)})")
        for a, b in (("Muon", "AdamW"), ("Muon", "SGD"), ("Muon", "Muon, p = 1")):
            ss = [s for s in SEEDS if weak.get((a, s)) is not None and weak.get((b, s)) is not None]
            if len(ss) >= 2:
                dd = [weak[a, s] - weak[b, s] for s in ss]
                print(f"  paired {a} - {b}: " + " ".join(f"s{s} {x:+.3f}" for s, x in zip(ss, dd))
                      + f"  mean {np.mean(dd):+.3f}  one-sided p = {p_one_sided(dd):.3f}")
        tally[lv] = dict(H1=(len(h1_ok), len(h1)), H1_vs_AdamW=(len(h1b_ok), len(h1b)), H2=(len(h2_ok), len(h2)),
                         H3=(len(h3_ok), len(h3)), weak={f"{k[0]}|{k[1]}": v for k, v in weak.items()},
                         rare={f"{k[0]}|{k[1]}": v for k, v in rare.items()},
                         common={f"{k[0]}|{k[1]}": v for k, v in common.items()})
    os.makedirs(os.path.join(ROOT, "figures/garg_scale"), exist_ok=True)
    json.dump(tally, open(os.path.join(ROOT, "figures/garg_scale/seeds_summary.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
