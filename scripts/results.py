"""Every number quoted in the paper, from the logs in runs/, in the order the paper uses them.

  python scripts/results.py
"""
import math
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)
from icl.analysis import SEEDS, SHIFTS, at_excess, linattn, nearest, pcr_ridge_ratio, run  # noqa: E402
from icl.data import spectrum  # noqa: E402
from icl.model import GPT2  # noqa: E402

ARMS = ["SGD", "AdamW", "Muon", "Muon, p = 1", "Muon, p = 0"]
LEVELS = (0.05, 0.04, 0.03)


def header(title):
    print(f"\n== {title}")


def per_seed(values):
    return " ".join(f"{v:+.3f}" for v in values) + f"  (> 0 in {sum(v > 0 for v in values)}/{len(values)})"


def setup():
    header("Methods")
    lam = spectrum(20)
    print(f"strongest / weakest direction variance: {lam[0] / lam[-1]:.0f}; top two directions: "
          f"{lam[:2].sum() / lam.sum():.0%} of the variance")
    model = GPT2()
    total = sum(p.numel() for p in model.parameters())
    hidden = sum(p.numel() for n, p in model.named_parameters() if n.startswith("blocks.") and p.ndim == 2)
    print(f"GPT-2 parameters: {total / 1e6:.1f}M, of which {1 - hidden / total:.1%} are trained by AdamW under Muon")


def speed():
    header("3.1 First step at each matched error (seeds 10-14)")
    for level in LEVELS:
        cells = []
        for arm in ARMS:
            steps = [run(arm, s).first_step(level) for s in SEEDS]
            if None not in steps:
                cells.append(f"{arm} {np.mean(steps):.0f} ({','.join(map(str, steps))})")
        print(f"  error {level}: " + "   ".join(cells))


def paired_t(d):
    """Paired t-test across seeds: the t statistic and its one-sided p-value (Student t, n - 1 degrees of freedom)."""
    d = np.asarray(d, dtype=float)
    k = len(d) - 1
    t = d.mean() / (d.std(ddof=1) / math.sqrt(len(d)))
    x = np.linspace(abs(t), abs(t) + 1000, 1_000_001)
    density = math.gamma((k + 1) / 2) / (math.sqrt(k * math.pi) * math.gamma(k / 2)) * (1 + x * x / k) ** (-(k + 1) / 2)
    return t, float(np.trapezoid(density, x))


def scores(arm, seed, level):
    """At the matched error: probe score and gain averaged over directions 3-20, and distances to the nearest
    reference of each family."""
    r = run(arm, seed)
    f = r.at(r.probe, level)
    if f is None:
        return None
    dist = {k: r.at(v, level) for k, v in r.dist.items()}
    return dict(probe=f[2:].mean(), gain=r.gain_at(level)[2:].mean(), learned=int((f > 0.5).sum()), f=f,
                r2=r.at(r.gain_r2, level, r.gain_rows),
                pcr=nearest(dist, "pcr_"), ridge=nearest(dist, "ridge_"), gd=min(nearest(dist, "gdpre_"), dist["gd1"]),
                nearest_m=min((k for k in dist if k.startswith("pcr_")), key=dist.get)[4:],
                nearest_a=min((k for k in dist if k.startswith("ridge_")), key=dist.get)[6:])


def algorithm():
    S = {(arm, s, lv): scores(arm, s, lv) for arm in ARMS for s in SEEDS for lv in LEVELS}
    keys = (("probe", "probe score"), ("gain", "gain"), ("pcr", "distance to PCR"), ("ridge", "distance to ridge"))
    header("3.2-3.3 and Table 1, at error 0.04: mean +- s.d. over seeds (probe score and gain: directions 3-20)")
    for arm in ARMS:
        v = lambda k: np.array([S[arm, s, 0.04][k] for s in SEEDS])
        print(f"  {arm:12s} " + "  ".join(f"{name} {v(k).mean():.3f} +- {v(k).std(ddof=1):.3f}" for k, name in keys)
              + f"  | learned (f > 0.5) {sorted(set(v('learned').tolist()))}, nearest PCR_m m = "
              f"{sorted(set(v('nearest_m').tolist()), key=int)}, ridge alpha = {sorted(set(v('nearest_a').tolist()))}")
    for a, b in (("Muon", "AdamW"), ("Muon, p = 0", "Muon, p = 1"), ("Muon", "SGD")):
        for k, name in keys:
            d = [S[a, s, 0.04][k] - S[b, s, 0.04][k] for s in SEEDS]
            print(f"  {a} - {b}, {name}: mean {np.mean(d):+.3f}; per seed {per_seed(d)}")
    header("Errors 0.05, 0.04 and 0.03 together (seed-error pairs)")
    for a, b in (("Muon", "AdamW"), ("Muon, p = 0", "Muon, p = 1"), ("Muon", "SGD")):
        pairs = [(S[a, s, lv], S[b, s, lv]) for s in SEEDS for lv in LEVELS if S[a, s, lv] and S[b, s, lv]]
        print(f"  {a} vs {b}: probe score higher in {sum(x['probe'] > y['probe'] for x, y in pairs)}/{len(pairs)}, "
              f"farther from PCR in {sum(x['pcr'] > y['pcr'] for x, y in pairs)}/{len(pairs)}")
    for lv in LEVELS:
        cells = []
        for a, b in (("Muon", "AdamW"), ("Muon, p = 0", "Muon, p = 1")):
            d = [S[a, s, lv]["gain"] - S[b, s, lv]["gain"] for s in SEEDS if S[a, s, lv] and S[b, s, lv]]
            cells.append(f"{a} - {b} gain {np.mean(d):+.3f} (> 0 in {sum(x > 0 for x in d)}/{len(d)})")
        print(f"  error {lv}: " + "   ".join(cells))
    for arm in ARMS:
        reads = [S[arm, s, lv] for s in SEEDS for lv in LEVELS if S[arm, s, lv]]
        print(f"  {arm:12s} closer to ridge than to PCR in {sum(x['pcr'] > x['ridge'] for x in reads)}/{len(reads)}")
    lead = np.mean([S["Muon", s, 0.04]["f"] - S["AdamW", s, 0.04]["f"] for s in SEEDS], 0)
    print(f"  Muon - AdamW probe score, summed over directions 12-15: {lead[11:15].sum():+.2f}; over the other "
          f"directions 3-20: {lead[2:11].sum() + lead[15:].sum():+.2f}")
    print(f"  linear fits behind the gains: R^2 >= {min(x['r2'] for x in S.values() if x):.4f} at every matched read")
    far = [S[arm, s, 0.04]["gd"] / min(S[arm, s, 0.04]["pcr"], S[arm, s, 0.04]["ridge"]) for arm in ARMS for s in SEEDS]
    print(f"  one-step gradient descent (nearest variant) is {min(far):.0f}-{max(far):.0f} times farther than PCR or ridge")


def shifts(level=0.04):
    header(f"3.4 Shifted queries at error {level}")
    q = {(arm, s): run(arm, s).shift_at(level) for arm in ARMS for s in SEEDS}
    print("  error / AdamW's, range over seeds, for each shift t (t < 0: common-ward, t > 0: rare-ward)")
    for arm in ("Muon", "Muon, p = 0", "Muon, p = 1", "SGD"):
        ratio = np.array([q[arm, s] / q["AdamW", s] for s in SEEDS])
        print(f"  {arm:12s} " + "  ".join(f"t={t:+g}: {ratio[:, i].min():.2f}-{ratio[:, i].max():.2f}"
                                           f" (lower in {(ratio[:, i] < 1).sum()}/{len(SEEDS)})"
                                           for i, t in sorted(enumerate(SHIFTS), key=lambda x: x[1]) if t != 0))
    for tt in (-1.0, 1.0):
        i = SHIFTS.index(tt)
        base = np.mean([q["AdamW", s][i] for s in SEEDS])
        d = [q["Muon", s][i] - q["AdamW", s][i] for s in SEEDS]
        t, p = paired_t(d)
        print(f"  t = {tt:+g}: AdamW's error {base:.3f}; Muon - AdamW {np.mean(d):+.4f} "
              f"({np.mean(d) / base:+.0%}), paired t = {t:+.1f}, one-sided p = {p:.3f}")


def linear_attention():
    header("3.5 Linear attention at excess loss 0.1: PCR fit / ridge fit, and rare-ward loss / Adam's (same seed); "
           "median [min-max] over seeds")
    for path in ("runs/linattn.json", "runs/linattn_rotated.json"):
        lam, runs = linattn(path)
        print(f"  {path}")
        by = {(r["arm"], r["seed"]): r for r in runs}
        for a in dict.fromkeys(r["arm"] for r in runs):
            seeds = sorted(s for arm, s in by if arm == a)
            ratio = [pcr_ridge_ratio(by[a, s], lam) for s in seeds]
            rare = [at_excess(by[a, s], by[a, s]["qloss"], 0.1) / at_excess(by["adam", s], by["adam", s]["qloss"], 0.1)
                    for s in seeds]
            print(f"    {a:10s} lr {by[a, seeds[0]]['lr']:<6g} PCR fit / ridge fit {np.median(ratio):.2f} "
                  f"[{min(ratio):.2f}-{max(ratio):.2f}]   rare-ward loss / Adam's {np.median(rare):.2f} "
                  f"[{min(rare):.2f}-{max(rare):.2f}]")


def lr_check():
    header("Appendix B: learning-rate check (seed 10, training loss over steps 1,400-1,500)")
    grids = {  # the rate used in the main runs, and the logs of every rate tried
        "SGD": (0.1, {0.03: "garg_lr/sgd_lr0.03_s10", 0.1: "garg_scale/sgd_s10", 0.3: "garg_lr/sgd_lr0.3_s10"}),
        "AdamW": (3e-4, {1e-4: "garg_lr/adamw_lr1e-4_s10", 3e-4: "garg_scale/adamw_s10",
                         5e-4: "garg_lr/adamw_lr5e-4_s10", 1e-3: "garg_lr/adamw_lr1e-3_s10"}),
        "Muon": (0.007, {0.0025: "garg_lr/muon_lr0.0025_s10", 0.007: "garg_scale/muon_s10",
                         0.02: "garg_lr/muon_lr0.02_s10"}),
        "Muon, p = 1": (5e-4, {1.5e-4: "garg_lr/muon_pow1_lr0.00015_s10", 5e-4: "garg_scale/muon_pow1_lr0.0005_s10",
                               1.5e-3: "garg_lr/muon_pow1_lr0.0015_s10", 7e-3: "garg_lr/muon_pow1_lr0.007_s10"}),
    }
    probe = {}
    for arm, (used, logs) in grids.items():
        loss = {}
        for lr, path in logs.items():
            r = run(path)
            loss[lr] = r.train_loss[np.isin(r.steps, (1450, 1500))].mean()
            if r.first_step(0.05) is not None and r.first_step(0.05) <= 1500:
                probe[arm, lr] = r.at(r.probe, 0.05)[2:].mean()
        best = min(loss, key=loss.get)
        print(f"  {arm:12s} " + "  ".join(f"{lr:g}: {v:.4f}" for lr, v in loss.items())
              + f"   used {used:g} is {loss[used] / loss[best] - 1:.1%} above the best ({best:g})")
    pairs = [(m, a) for (arm_m, m) in probe if arm_m == "Muon" for (arm_a, a) in probe if arm_a == "AdamW"]
    wins = sum(probe["Muon", m] > probe["AdamW", a] for m, a in pairs)
    print(f"  Muon's probe score at error 0.05 exceeds AdamW's in {wins}/{len(pairs)} pairs of rates")


if __name__ == "__main__":
    setup()
    speed()
    algorithm()
    shifts()
    linear_attention()
    lr_check()
