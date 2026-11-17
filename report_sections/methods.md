## Methods (pilot and verification)

**Task (as proposed).** d = 8, prompts of 32 pairs (x_i, wᵀx_i) plus a query, fresh w ~ N(0, I) for
every prompt, noiseless; x ~ N(0, RΛRᵀ) with λ_i ∝ 1, 1/i, 1/i² rescaled to trace 8; R = I
("aligned") or one fixed Haar rotation ("rotated"). Every prefix k = 0..32 is trained and scored
(Garg-style), errors normalised so that predicting 0 scores 1. The noiseless optimum is least
squares (min-norm; it is also the Bayes estimator), which is exact once k ≥ 8.

**Model (as proposed).** 4-layer, 4-head, 128-d GPT-2 matching Garg et al.'s `TransformerModel`
(pre-LN, learned positions, 4× gelu MLP, no dropout, linear read-in/read-out), 0.80 M parameters.

**Optimizers.** SGD with momentum 0.9; AdamW (β = 0.9, 0.999); Muon from Keller Jordan's reference
(NS5 coefficients 3.4445/−4.7750/2.0315, Nesterov momentum 0.95, √max(1, rows/cols) scaling) on
the 24 hidden matrices, with AdamW (lr 1e-3) on read-in, read-out, positional embeddings, biases
and LayerNorm. Q, K and V are separate matrices so Muon orthogonalises each. Weight decay 0 for
all, 100-step warm-up, then a constant LR. Batch 64.

**Two theory checks added to the proposal's design:**
1. *A linear-attention testbed.* One-layer, 8-head linear attention (Zhang et al.'s separate
   rank-1 K/Q and merged-KQ variants, small init) on the same data, trained with GD, Adam,
   SignGD, Muon (NS5) and Muon with the exact polar factor, each with its own 5-point LR sweep and
   3 seeds. This is the setting the cited theory (Zhang et al., Ma et al.) is actually about;
   each run takes seconds, so it tests the hypotheses' premises cleanly before the GPT-2 runs ask
   whether they transfer.
2. *Implied-weight gains.* For each of 64 fixed contexts the GPT-2 is queried at 16 in-distribution
   queries and a linear map ŷ ≈ ŵᵀx_q + b is fitted (R² checks linearity). The gain on principal
   axis i, ρ_i = slope of u_iᵀŵ on u_iᵀw, is the model's per-direction "filter": all 1 = least
   squares, a 0/1 step = PCR, a smooth λ-dependent profile = ridge-like. This replaces probe
   tasks w ∝ u_i, whose y-scale is out of distribution for weak axes.

**Verification additions.**
- *Reproductions* (`icl/repro_theory.py`, `icl/linattn.py`, `scripts/garg_eval.py`):
  - Zhang et al.'s gradient flow, integrated with an adaptive ODE solver (LSODA, rtol 1e-7).
  - Ma et al.'s quadratic f(Q) on a common step-size grid for every method.
  - Deep linear networks, with one LR per method chosen on the aligned problem.
  - Garg et al.'s full evaluation suite (in-distribution, skewed covariance, noisy labels,
    half-dimensional subspace, scaled inputs, orthogonal and overlapping queries) with least
    squares, 3-NN, averaging and Bayes-ridge baselines.
  - Garg et al.'s curriculum (`--curriculum garg`: active dimensions 2 → 8, 4·dims + 1 points).
  - Xie et al.'s rotated parameter space (`--param_rotation`): Adam runs on φ with θ = Rφ, where R
    is two random permutations around a Kronecker product of three Haar rotations; RᵀR = I to
    3e-6.
- *Hyperparameter search* (`scripts/search_report.py`):
  - Per condition (isotropic and λ ∝ 1/i² aligned), ranked by area under the test-error curve over
    a fixed budget (AdamW and SGD 4k steps, Muon 3k).
  - Scored on the pilot's prompt set (seed 20000), which then counts as a selection set.
  - SGD fairness checks: LR 0.01–1.0, clipping, d = 2, and the curriculum.
- *Confirmation runs* (`scripts/confirm_jobs.py`, `scripts/analyze_confirm.py`,
  `scripts/make_confirm_figures.sh`):
  - Fresh seeds 10–12, evaluated on a held-out prompt set (seed 30000).
  - A constant-LR phase (AdamW 8k, Muon 3k), then a linear decay to 0 over 2k / 1k steps,
    branched from a checkpoint. Resuming is bit-exact (checked against an uninterrupted run).
  - Dense evaluation (every 25 steps up to 2k, then every 100). The query-shift test runs every
    500 steps and the implied-weight fit every 250.
- *Numerics check*: one seed per optimizer retrained fully in fp32.

**Deviations from the proposal, and why.**

| Proposal | What was run | Reason |
|---|---|---|
| Garg's token layout (y in coordinate 0 of the x read-in) | y in its own (d+1)-th input coordinate; Gaussian read-in init | Garg's layout makes the y coordinate coincide with the top-variance axis when R = I, confounding aligned vs rotated. With the fix, SGD and read-in-Muon are *exactly* rotation-equivariant (verified: aligned vs rotated loss trajectories agree to 1e-6), a free positive control. |
| 3 optimizers × 5 covariances × 3 seeds, 30 GPU-h | CPU only (4 cores, one run per core, ~0.22 s/step). Pilot: AdamW and Muon on 4 covariances, 1–2 seeds. Confirmation: AdamW and Muon on isotropic, 1/i² aligned and 1/i² rotated, 3 seeds (2 for Muon rotated). SGD only in the fairness checks, since it never leaves the plateau. | No GPU in the sandbox. |
| Training length (Garg: 500k steps) | AdamW 10k, Muon 4k steps, each ending in an LR decay | Budget. Muon converges by ~1.4k steps. AdamW converges by 5–6k on average (3.7–7.7k across seeds), though not on every axis. |
| LR sweep per condition | Pilot: isotropic only. Verification: per condition (isotropic and 1/i²; rotated reuses the aligned winner) | The pilot's shortcut under-tuned both optimizers on 1/i². |
| 2,000 test prompts | 1,000 test + 512 probe + 1,280 query-shift + 1,024 implied-weight prompts per evaluation set; separate selection (20000) and held-out (30000) sets | Budget; tuning and test prompts kept separate. |
