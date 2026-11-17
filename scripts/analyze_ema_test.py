"""Raw vs time-averaged (EMA) predictor: excess query-shift loss at matched in-distribution error.

  python scripts/analyze_ema_test.py runs/linattn/ema_test.json
"""
import json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from analyze_linattn import metrics  # noqa: E402

LEVELS = (0.1, 0.05)


def q_at(m, lv):
    exc, q = m["exc"], m["q"]
    i = np.nonzero(exc <= lv)[0]
    if not len(i):
        return np.nan
    i = int(i[0])
    if i == 0:
        return float(q[0])
    t = (exc[i - 1] - lv) / (exc[i - 1] - exc[i])
    return float(q[i - 1] + t * (q[i] - q[i - 1]))


def rho_at(m, lv):
    exc, rho = m["exc"], m["rho"]
    i = np.nonzero(exc <= lv)[0]
    return None if not len(i) else rho[int(i[0])]


def main():
    R = json.load(open(sys.argv[1] if len(sys.argv) > 1 else "runs/linattn/ema_test.json"))
    out = {}
    for cond in ("1/i2-ali", "1/i-ali"):
        RC = [r for r in R if r["cond"] == cond]
        M = [(r, metrics(r), metrics(dict(r, loss=r["loss_ema"], qloss=r["qloss_ema"], rho=r["rho_ema"]))) for r in RC]
        floor = min(min(float(m["q"][-1]), float(me["q"][-1])) for _, m, me in M)
        print(f"\n== {cond}: excess query-shift loss at matched in-distribution excess (median of 3 seeds), raw -> averaged; floor {floor:.3f}")
        for o in ("adam", "muon", "muon_exact"):
            for b in (16, 64, 256, 1024):
                for ema in (0.95, 0.99):
                    v = [(m, me) for r, m, me in M if r["opt"] == o and r["batch"] == b and r["ema"] == ema]
                    if not v:
                        continue
                    raw = [np.nanmedian([q_at(m, lv) - floor for m, _ in v]) for lv in LEVELS]
                    avg = [np.nanmedian([q_at(me, lv) - floor for _, me in v]) for lv in LEVELS]
                    prof = [rho_at(me, 0.1) for _, me in v]
                    prof = np.median([p for p in prof if p is not None], 0) if any(p is not None for p in prof) else None
                    out[f"{cond}|{o}|{b}|{ema}"] = dict(raw=raw, avg=avg, profile=None if prof is None else prof.tolist())
                    f = lambda x: "  -  " if x != x else f"{x:.3f}"
                    print(f"  {o:10s} b{b:<5d} ema {ema}: @0.1 {f(raw[0])} -> {f(avg[0])}   @0.05 {f(raw[1])} -> {f(avg[1])}   "
                          f"averaged gains @0.1: {'-' if prof is None else ' '.join(f'{x:4.2f}' for x in prof)}")
    json.dump(out, open("figures/noise/ema_test.json", "w"), indent=1, default=float)


if __name__ == "__main__":
    main()
