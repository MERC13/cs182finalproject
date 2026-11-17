"""Does the initialisation scale decide whether optimizers differ in WHAT the GPT-2 learns, or only how fast?

  python scripts/analyze_init.py

For each run (λ ∝ 1/i², aligned, batch 64): steps at which the 8 directions are learned (probe error < 0.5),
the spectral exponent beta, and the fraction of each direction learned (1 - probe error) at matched
in-distribution error (32 examples) 0.1 and 0.03. Same-shape profiles at matched error = the optimizers
differ only in speed; different shapes = they learn different in-context predictors.
"""
import glob, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from icl.analysis import load  # noqa: E402
from analyze_noise import beta  # noqa: E402

GROUPS = [("original GPT-2 (separate tokens, init 0.02)", "runs/confirm/*_1i2-ali_s10.json"),
          ("concatenated tokens, init 0.02", "runs/init/*_init002_s10.json"),
          ("concatenated tokens, init 0.044", "runs/init/*_init044_s1[01].json"),
          ("concatenated tokens, init 0.088", "runs/rescue/*_1i2-ali_s10.json")]


def profile(r, lv):
    i = np.nonzero(r.err_last <= lv)[0]
    return (None, None) if not len(i) else (int(r.steps[int(i[0])]), 1 - r.probe_err()[int(i[0])])


def main():
    for title, pat in GROUPS:
        files = [f for f in sorted(glob.glob(pat)) if "decay" not in f]
        if not files:
            continue
        print(f"\n== {title}")
        for f in files:
            r = load(f)
            T = r.axis_learn_steps(0.5)
            b = beta(T, r.lam)
            name = os.path.basename(f).replace(".json", "")
            print(f"  {name:28s} {r.status:8s} axes {[None if not np.isfinite(t) else int(t) for t in T]}  beta {b:.2f}")
            for lv in (0.1, 0.03):
                st, p = profile(r, lv)
                print(f"      at error {lv:<5}: " + ("never" if p is None else f"step {st:5d}  " + " ".join(f"{x:5.2f}" for x in p)))


if __name__ == "__main__":
    main()
