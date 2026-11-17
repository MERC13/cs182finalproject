## Verification step 2 — is each independent variable tested without confounds?

**Optimizer: each optimizer is tuned per condition, not borrowed from the isotropic sweep.**
The search ran on the isotropic and λ ∝ 1/i² aligned conditions, ranked by area under the
test-error curve over a fixed budget (4k steps for AdamW and SGD, 3k for Muon). It used
selection prompts kept separate from the held-out prompts used for the final numbers.

| | Isotropic | λ ∝ 1/i² aligned (also used for rotated) |
|---|---|---|
| AdamW, LR | 1e-4 … 1e-3 (β2 0.999) and 3e-4 (β2 0.95): best **5e-4, β2 0.999** (AUC 0.596 vs 0.696 for the pilot's 3e-4); 1e-3 is much worse (0.907) and 3e-3 never escapes | 1e-4 … 1e-3 (β2 0.999) and 3e-4 … 5e-4 (β2 0.95): best **5e-4, β2 0.999** (AUC 0.255 vs 0.291 for the pilot's 3e-4). β2 = 0.95 made no real difference. |
| Muon, LR | 0.0025 … 0.02: best **0.005** (AUC 0.292), the pilot's value; 0.02 and 0.05 never escape | 0.002 … 0.01: best **0.007** (AUC 0.158, tied with 0.01; lower final error); the pilot's 0.005 scored 0.171 |
| SGD + momentum 0.9 | lr 0.01 and 0.03 (no clipping, 4k steps); 0.1 without clipping diverges by step 42; 0.1 with gradient clipping at norm 1 (10k steps); 1.0 with clipping is unstable (error 20.5 at step 2k, 8.6 at 3k, worse than predicting 0). **No setting leaves the plateau.** Momentum 0.99 was not tested. | not run (SGD never learns at all) |

The pilot's isotropic-tuned LRs were too small for both adaptive optimizers on the anisotropic
inputs. The confirmation runs use the per-condition winners.

**SGD gets a fair chance and still fails, for an identifiable reason.**
- It fails at d = 2 (4k steps), where AdamW learns within 500 steps.
- It fails with Garg et al.'s curriculum (still at error 1.000 after 5k steps).
- It fails at every LR / momentum / clipping setting above.
- The cause is measured directly. On the plateau the query/key gradient norm is 9×10⁻⁵ against a
  total gradient norm of 0.29, dominated by the read-out layer, and after 500 steps the
  query/key weights have not moved (norm 7.245 → 7.245). With one global learning rate, capped
  by the read-out's curvature, the attention circuit never starts to form. Per-coordinate
  (AdamW) or per-matrix (Muon) normalisation gives query/key learning-rate-sized steps
  regardless.
- So SGD's failure is an optimizer × architecture effect (heterogeneous gradient scales across
  blocks). It is not the input-variance effect H2 is about, and it means H2/H4 cannot be tested
  with plain SGD in this model.

![SGD fairness checks](figures/verify/sgd_fairness.png)

**Numerics: bf16 autocast is not a confound.** Training the same seed and config fully in fp32 gives identical plateau-escape steps (AdamW 1,900 vs 1,900; Muon 400 vs 400). The test-error curves differ by at most 0.020 (AdamW) and 0.003 (Muon) over the compared steps.

**Training length: Garg et al.'s curriculum and longer training.**
- Under Garg's own recipe (Adam 1e-4, batch 64), the curriculum reproduces their speed-up:
  error 0.204 at step 8k vs 0.356 without it.
- Every confirmation run ends with a linear learning-rate decay branch, resumed from the
  constant-LR checkpoint. End-state claims (final error, query shift, skewed-covariance
  robustness) therefore compare models after an LR decay, not a constant-LR noise floor. Even
  so, 2 of 6 AdamW 1/i² runs had still not learned the weakest axis at the end (step 3).

**Covariance alignment: the instrument can detect basis effects.** Rotating AdamW's *entire parameter space* with Xie et al.'s construction (two random permutations around a Kronecker product of three Haar rotations) is decisive. With the same seed, LR (3e-4) and data, rotated AdamW never leaves the plateau in 6k steps (validation error 0.995), while standard AdamW escapes at step 1,900. So this GPT-2 is highly sensitive to Adam's coordinate system, and the null effect of rotating only the *input covariance* (step 3) is a real negative result, not an insensitive instrument. Input rotation only reaches the read-in's input side; the hidden layers' coordinates are already random with respect to the data's principal axes.

![Xie et al. check](figures/verify/repro_xie.png)
