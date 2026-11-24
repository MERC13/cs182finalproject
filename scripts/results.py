"""Every number quoted in the paper, from the logs in runs/, in the order the paper uses them.

  python scripts/results.py
"""
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)
from icl.analysis import SEEDS, SHIFTS, at_excess, linattn, nearest, pcr_ridge_ratio, run  # noqa: E402
from icl.data import spectrum  # noqa: E402
from icl.model import GPT2  # noqa: E402

ARMS = ["SGD", "AdamW", "Muon", "Muon, p = 1"]
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


def scores(arm, seed, level):
    """Probe profile, raw and bounded score over directions 3-20, and distances to the reference families, all read
    at the matched error."""
    r = run(arm, seed)
    f = r.at(r.probe, level)
    if f is None:
        return None
    dist = {k: r.at(v, level) for k, v in r.dist.items()}
    return dict(f=f, raw=f[2:].mean(), bounded=np.clip(f[2:], 0, None).mean(), learned=int((f > 0.5).sum()),
                pcr=nearest(dist, "pcr_"), ridge=nearest(dist, "ridge_"),
                gd=min(nearest(dist, "gdpre_"), dist["gd1"]))


def algorithm():
    S = {(arm, s, lv): scores(arm, s, lv) for arm in ARMS for s in SEEDS for lv in LEVELS}
    header("3.2 and Table 1, at error 0.04: mean +- s.d. over seeds")
    for arm in ARMS:
        v = lambda k: np.array([S[arm, s, 0.04][k] for s in SEEDS])
        print(f"  {arm:12s} directions learned {sorted(set(v('learned').tolist()))}  probe {v('raw').mean():.2f} +- "
              f"{v('raw').std(ddof=1):.2f}  bounded {v('bounded').mean():.2f} +- {v('bounded').std(ddof=1):.2f}  "
              f"distance to PCR {v('pcr').mean():.3f}, to ridge {v('ridge').mean():.3f}")
    for key, name in (("raw", "probe score"), ("bounded", "bounded score"), ("pcr", "distance to PCR")):
        d = [S["Muon", s, 0.04][key] - S["AdamW", s, 0.04][key] for s in SEEDS]
        print(f"  Muon - AdamW, {name}: mean {np.mean(d):+.3f}; per seed {per_seed(d)}")
    for other in ("AdamW", "SGD", "Muon, p = 1"):
        pairs = [(S["Muon", s, lv], S[other, s, lv]) for s in SEEDS for lv in LEVELS if S[other, s, lv]]
        n = sum(a["pcr"] > b["pcr"] for a, b in pairs)
        print(f"  Muon farther from PCR than {other} in {n}/{len(pairs)} seed-error pairs "
              f"(errors {', '.join(map(str, LEVELS))})")
    far = [S[arm, s, 0.04]["gd"] / min(S[arm, s, 0.04]["pcr"], S[arm, s, 0.04]["ridge"]) for arm in ARMS for s in SEEDS]
    print(f"  one-step gradient descent (nearest variant) is {min(far):.0f}-{max(far):.0f} times farther than PCR or ridge")
    header("Same comparisons at errors 0.05 and 0.03")
    for lv in (0.05, 0.03):
        for other in ("AdamW", "SGD", "Muon, p = 1"):
            d = [S["Muon", s, lv]["raw"] - S[other, s, lv]["raw"] for s in SEEDS if S[other, s, lv]]
            if d:
                print(f"  error {lv}: Muon - {other} probe score {per_seed(d)}")


def shifts(level=0.04):
    header(f"3.3 Shifted queries at error {level}")
    q = {(arm, s): run(arm, s).shift_at(level) for arm in ARMS for s in SEEDS}
    print("  error / AdamW's, range over seeds, for each shift t (t < 0: common-ward, t > 0: rare-ward)")
    for arm in ("Muon", "Muon, p = 1", "SGD"):
        ratio = np.array([q[arm, s] / q["AdamW", s] for s in SEEDS])
        print(f"  {arm:12s} " + "  ".join(f"t={t:+g}: {ratio[:, i].min():.2f}-{ratio[:, i].max():.2f}"
                                           for i, t in sorted(enumerate(SHIFTS), key=lambda x: x[1]) if t != 0))
    for tt in (-1.0, 1.0):
        i = SHIFTS.index(tt)
        base = np.mean([q["AdamW", s][i] for s in SEEDS])
        diff = np.mean([q["Muon", s][i] - q["AdamW", s][i] for s in SEEDS])
        print(f"  t = {tt:+g}: AdamW's error {base:.3f}, Muon - AdamW {diff:+.3f} (mean over seeds)")


def exact_muon():
    header("3.4 Exact orthogonalization (seed 10) next to Muon and AdamW: probe score at each error")
    for lv in LEVELS:
        cells = []
        for name in ("muon_exact_s10", "muon_s10", "adamw_s10"):
            r = run(f"garg_scale/{name}")
            cells.append(f"{name} {r.at(r.probe, lv)[2:].mean():+.3f}")
        print(f"  error {lv}: " + "  ".join(cells))


def linear_attention():
    header("3.4 Linear attention: PCR fit / ridge fit (median [min-max] over seeds); rare-ward loss vs Adam at excess 0.1")
    lam, runs = linattn()
    arms = list(dict.fromkeys(r["arm"] for r in runs))
    rare = {a: np.median([at_excess(r, r["qloss"], 0.1) for r in runs if r["arm"] == a]) for a in arms}
    for a in arms:
        ratios = [pcr_ridge_ratio(r, lam) for r in runs if r["arm"] == a]
        lr = next(r["lr"] for r in runs if r["arm"] == a)
        print(f"  {a:10s} lr {lr:<6g} ratio {np.median(ratios):.2f} [{min(ratios):.2f}-{max(ratios):.2f}]  "
              f"rare-ward loss / Adam's {rare[a] / rare['adam']:.2f}")


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
    exact_muon()
    linear_attention()
    lr_check()
