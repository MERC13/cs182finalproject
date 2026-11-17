"""Zipfian task pool: does Muon memorise rare tasks relatively sooner than AdamW? (Wang et al. 2025, arXiv 2509.26030)

  python scripts/analyze_zipf.py 'runs/zipf/*.json'

At matched memorisation score m on the training distribution (= matched training loss), compare the
memorisation scores of the most frequent eighth of the tasks (head) and the least frequent half (tail),
and the ICL score s on unseen tasks. Same definitions as scripts/analyze_transience.py.
"""
import glob, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from analyze_transience import load_runs, smooth  # noqa: E402


def at_m(c, key, level):
    m = c["m_sm"]; v = smooth(c[key]) if key in ("m_head", "m_tail") else c[key + "_sm"]
    i = np.nonzero(m >= level)[0]
    if not len(i):
        return None
    i = int(i[0])
    if i == 0:
        return float(v[0])
    t = (level - m[i - 1]) / (m[i] - m[i - 1])
    return float(v[i - 1] + t * (v[i] - v[i - 1]))


def main():
    runs = load_runs(sys.argv[1:] or ["runs/zipf/*.json"])
    for R in runs:
        c = R["c"]
        print(f"== {R['opt']} M={R['M']} seed {R['seed']} ({R['status']}, to step {int(c['step'][-1])}); final m {c['m'][-1]:.2f} "
              f"head {c['m_head'][-1]:.2f} tail {c['m_tail'][-1]:.2f} s {c['s'][-1]:.2f}")
        for lv in (-0.5, 0.0, 0.25, 0.5, 0.75):
            h, t, s = at_m(c, "m_head", lv), at_m(c, "m_tail", lv), at_m(c, "s", lv)
            i = np.nonzero(c["m_sm"] >= lv)[0]
            if h is None:
                continue
            print(f"   at m = {lv:+.2f} (step {int(c['step'][i[0]]):6d}): head {h:+.2f}  tail {t:+.2f}  tail - head {t - h:+.2f}  ICL s {s:.2f}")


if __name__ == "__main__":
    main()
