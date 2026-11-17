## Methodological findings (fix these before the full run)

1. **Garg's token layout breaks the symmetry the aligned-vs-rotated comparison relies on.** We showed this is real but did not measure its practical size; AdamW's rotation effect in this GPT-2 is null anyway (step 3). Garg's code writes y into
   coordinate 0 of the same read-in as x. With R = I that coordinate is also the top-variance input
   axis; under a rotation it is not. Giving y its own (d+1)-th coordinate, with an iid-Gaussian
   read-in init, makes SGD — and Muon if it also updates the read-in — **exactly**
   rotation-equivariant. We checked this pathwise: with matched seeds, aligned and rotated
   training losses agree to 2e-6 (SGD) and 1e-6 (Muon incl. read-in) over 40 steps, against 2.5e-2
   for AdamW and 1.0e-1 for standard Muon, whose read-in uses AdamW. Consequences:
   - The rotated condition is a free replicate for SGD, i.e. a positive control.
   - Only AdamW, and the AdamW-updated parts of Muon, can tell aligned from rotated. Muon with an
     AdamW read-in is *not* rotation-invariant.
   - Rotating only the inputs touches only the read-in's input side. Expect a much smaller Adam
     effect than Xie et al.'s full parameter rotation.
2. **The ICL plateau is long and LR-sensitive, so tune per condition.**
   - On isotropic inputs (one seed per LR, 3–4k steps) the adaptive optimizers stall completely
     above a threshold: AdamW at 3e-3, Muon at 0.02 and 0.05.
   - Below the threshold, escape time varies non-monotonically with LR (AdamW: 1,200–2,400 steps
     across 1e-4 … 1e-3). Part of this is seed noise: three fresh seeds at one LR spread
     1,075–1,300.
   - The best LR also differs by condition. The pilot's isotropic choices were too small for
     λ ∝ 1/i²: AdamW 3e-4 → 5e-4, and Muon 0.005 → 0.007.
   - Sweep a decade below the usual defaults (Muon's reference LR is 0.02), select by area under
     the learning curve on separate prompts, and tune each condition.
   - **Expect large seed-to-seed variance for AdamW on skewed spectra.** One of three seeds had not
     learned the weakest axis after 8k steps, so budget ≥ 3 seeds.

   ![LR sweep](figures/grid/lr_sweep_iso.png)
3. **Garg et al.'s curriculum cannot simply be reused.** Its dimension curriculum zeroes raw
   coordinates, which distorts rotated spectra. Zeroing the low-variance axes first would bake the
   effect being tested into the data. A points-only curriculum is safe.
4. **Reference estimators need care in the noiseless setting.** Ridge(λ→0) = least squares =
   the Bayes estimator here, so "ridge-like" has to mean a fitted λ > 0 (we report the best-fitting
   λ per checkpoint). Plain least squares restricted to the top-m axes explodes near k = m
   (normalised error 618 at k = 1, m = 1). Use either the Bayes version (subspace ridge with
   penalty Σ_{j>m} λ_j) or Zhang et al.'s finite-N form, which is one-step preconditioned GD with
   c_i = N/((N+1)λ_i + trΛ). The latter matches its analytic error (0.1619 vs 0.1610 on
   200k prompts).
5. **Probe tasks w ∝ u_k are out of distribution in scale** for weak axes (Var y = dλ_k), and
   64 probes per axis give ±0.18 noise per position. Ratio-normalised errors and the
   implied-weight gains (fit ŷ ≈ ŵᵀx_q per context) are better per-direction measures. An OLS
   oracle recovers gain 1 and error 1e-15 exactly.
6. **Muon details matter for the weak directions.** NS5 does not equalise singular values below
   ~1e-3 of the Frobenius norm. In the testbed the weakest axis takes 160 steps with NS5 vs 90
   with the exact polar factor. With a constant LR, Muon also keeps a noise floor. Run an exact-SVD
   ablation and use an LR schedule.
7. **Compute.** The 0.80M-parameter model takes 0.22 s/step on one CPU core (bf16 linears, fp32
   attention). A GPU should be far faster per step; measure first, as the proposal says.
