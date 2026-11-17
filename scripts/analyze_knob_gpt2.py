"""Spectral-flattening knob in the redesigned GPT-2 (concatenated tokens, hidden init 0.044, λ ∝ 1/i², seed 10).

  python scripts/analyze_knob_gpt2.py

Rows: SGD, AdamW and standard Muon from runs/init (seed 10), and Muon with the update's singular values raised
to p = 1, 0.25, 0 from runs/knob (jobs/knob_gpt2.txt). Per run:
  learned   1 - probe error per direction at the first evaluation with in-distribution error at 32 examples <= 0.03;
            "weak" = its mean over directions 3-8 (the proposal's H1 statistic)
  q@0.03    shifted-query error at matched in-distribution error 0.03 (first crossing, log-linear interpolation;
            the knob runs evaluate it every 100 steps, the runs/init ones every 500)
  beta      learning-order exponent; T_conv first step with excess error below 0.05; final error
"""
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
from icl.analysis import load  # noqa: E402
from analyze_confirm import run_metrics  # noqa: E402
from analyze_noise import beta  # noqa: E402

RUNS = [("SGD", "runs/init/sgd_lr0.1_init044_s10.json"), ("AdamW", "runs/init/adamw_init044_s10.json"),
        ("Muon (Newton-Schulz)", "runs/init/muon_init044_s10.json"),
        ("p = 1", "runs/knob/muon_pow1_init044_s10.json"), ("p = 0.25", "runs/knob/muon_pow0.25_init044_s10.json"),
        ("p = 0", "runs/knob/muon_pow0_init044_s10.json")]


def main():
    print(f"{'arm':22s} {'status':8s} | {'learned at error 0.03 (directions 1-8)':47s} | {'weak':>5s} | "
          f"{'q@0.03':>6s} {'q@0.01':>6s} | {'beta':>5s} {'T_conv':>6s} {'final':>6s}")
    for name, f in RUNS:
        path = os.path.join(ROOT, f)
        if not os.path.exists(path):
            print(f"{name:22s} missing")
            continue
        r = load(path)
        m = run_metrics(r)
        i = np.nonzero(r.err_last <= 0.03)[0]
        prof = 1 - r.probe_err()[int(i[0])] if len(i) else np.full(8, np.nan)
        q = m["q1_matched"]
        T = r.axis_learn_steps(0.5)
        print(f"{name:22s} {r.status if hasattr(r, 'status') else '':8s} | " + " ".join(f"{v:5.2f}" for v in prof)
              + f" | {np.nanmean(prof[2:]):5.2f} | {q[0]:6.3f} {q[1]:6.3f} | {beta(T, r.lam):5.2f} {m['T_conv']:6.0f} "
              f"{float(r.err_mean[-1]):6.3f}")


if __name__ == "__main__":
    main()
