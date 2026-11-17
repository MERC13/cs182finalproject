## Verification step 3 — confirmation runs (GPT-2, tuned per condition, fresh seeds, held-out prompts)

Protocol, fixed before the runs:
- **Configs.** Per-condition tuned settings from step 2: AdamW 5e-4 (β2 0.999) everywhere; Muon
  0.005 on isotropic and 0.007 on λ ∝ 1/i² inputs (aux AdamW 1e-3).
- **Seeds and prompts.** Fresh seeds 10–12 (two seeds for Muon rotated), not used in any search.
  Evaluation on a held-out prompt set (seed 30000) that no search touched.
- **Schedule.** A constant-LR phase (AdamW 8k steps, Muon 3k), then a linear learning-rate decay
  to zero (2k / 1k steps), branched from the saved checkpoint. Timing metrics come from the
  constant-LR phase; end-state metrics ("after decay") from the end of the decay.
- **Evaluation frequency.** Every 25 steps up to step 2,000, then every 100; query-shift and
  implied-weight tests every 500 / 250 steps.

Values are mean [min–max] over seeds. "Never" means not reached within the constant-LR phase.
Definitions:
- **escape:** mean error first below 0.9.
- **converged:** excess error over least squares (averaged over k = 1..32) < 0.05.
- **weakest axis:** probe error on axis 8 first below 0.5.
- **PCR/ridge:** median, over the learning window (excess error between 5% and 90% of its
  starting value), of the distance from the model's predictions to the best PCR(m) divided by
  the distance to the best ridge(λ), for k = 8..32. Below 1 is PCR-like (staircase), above 1
  ridge-like.
- **shifted-query error:** same contexts and tasks, but the query is drawn from the inverted
  spectrum (λ_i⁻¹ rescaled to trace 8), so it loads the weak axes; error at the last position.
- **weakest-axis gain:** implied-weight gain on axis 8 (1 = learned like least squares, 0 = ignored).

| covariance | optimizer | seeds | escape | converged | weakest axis | axis-8 / axis-1 time | PCR/ridge | error @32 after decay | shifted-query error after decay | weakest-axis gain after decay |
|---|---|---|---|---|---|---|---|---|---|---|
| isotropic | AdamW | 3 | 1,167 [1,075–1,300] | 5,933 [3,700–7,700] | 1,550 [1,450–1,600] | 0.98 [0.88–1.08] | 1.84 [1.76–1.90] | 0.0054 [0.0034–0.0087] | 0.0062 [0.0029–0.0097] | 1.01 [1.00–1.02] |
| isotropic | Muon | 3 | 433 [400–450] | 1,392 [1,325–1,450] | 550 [525–575] | 0.96 [0.91–1.00] | 1.72 [1.69–1.74] | 0.0013 [0.0010–0.0020] | 0.0015 [0.0011–0.0022] | 1.00 [1.00–1.00] |
| 1/i² aligned | AdamW | 3 | 150 [125–175] | 5,250 [4,900–5,600] (+1 never) | 6,350 [6,200–6,500] (+1 never) | 37 [31–43] (+1 never) | 0.76 [0.67–0.84] | 0.014 [0.003–0.034] | 0.34 [0.14–0.66] | 0.61 [−0.01–1.00] |
| 1/i² aligned | Muon | 3 | 167 [150–200] | 867 [750–975] | 900 [750–1,025] | 4.3 [3.8–4.6] | 1.17 [1.11–1.25] | 0.0011 [0.0008–0.0015] | 0.040 [0.031–0.045] | 1.01 [1.00–1.01] |
| 1/i² rotated | AdamW | 3 | 183 [175–200] | 6,200 [4,900–6,900] | 6,400 (+2 never) | 32 (+2 never) | 0.80 [0.75–0.84] | 0.010 [0.003–0.022] | 0.27 [0.09–0.52] | 0.62 [0.00–0.94] |
| 1/i² rotated | Muon | 2 | 188 [175–200] | 888 [850–925] | 925 [875–975] | 4.4 [4.3–4.4] | 1.15 [1.14–1.17] | 0.0009 [0.0009–0.0009] | 0.031 [0.030–0.031] | 1.00 [1.00–1.01] |

The full table (half-way times, constant-LR end states) is in `figures/confirm/confirm_table.md`
and the per-seed values in `figures/confirm/confirm.json`.

* **H1 — confirmed; the isotropic effect is smaller than the pilot said.**
  - Isotropic: Muon leaves the plateau at 433 [400–450] steps vs 1,167 [1,075–1,300] for AdamW
    (2.7×). It converges at 1,392 [1,325–1,450] vs 5,933 [3,700–7,700] (4.3×). The pilot said ~5×.
  - λ ∝ 1/i²: converged at 867 vs 5,250 steps aligned (6×; one AdamW seed did not converge within
    8k) and 888 vs 6,200 rotated (7×).
  - The seed ranges never overlap. Muon also ends lower after the decay (error@32 0.0013 vs
    0.0054 isotropic, 0.0011 vs 0.014 under 1/i² aligned), although AdamW trained for 10k steps
    against Muon's 4k.
  - First escape is uninformative under 1/i² (150–190 steps for both): the top axis carries most
    of the variance and is learned at once. That is why convergence and weakest-axis times are
    the H1/H2 metrics.
* **H2 (with AdamW in SGD's place) — confirmed.**
  - AdamW's weakest axis takes 1,550 → 6,350 steps (isotropic → 1/i² aligned; one seed never in
    8k), and its last-axis/first-axis time ratio grows 1.0 → 37 [31–43].
  - Muon's weakest axis takes 550 → 900 steps, and its ratio grows only 1.0 → 4.3 [3.8–4.6].
  - Plain SGD cannot be scored: it never leaves the plateau (step 2).
* **H4 (revised: AdamW, not SGD, runs the PCR staircase) — confirmed at tuned LRs on fresh seeds.**
  - AdamW under 1/i² learns the axes one at a time, in eigenvalue order, in every seed (seed 10
    aligned: steps 200, 700, 1,525, 2,400, 3,500, 4,200, 4,800, 6,200).
  - Its predictions are closer to the best PCR(m) than to any ridge for 79–96% of the learning
    window. The PCR/ridge ratio is 0.76 [0.67–0.84] aligned and 0.80 [0.75–0.84] rotated, every
    seed below 1.
  - Muon's ratio is 1.17 [1.11–1.25] aligned and 1.15 rotated, every seed above 1. It is
    PCR-like for only 12–23% of its window.
  - Muon is not literally parallel. It also adds axes in eigenvalue order (the best-fitting PCR
    rank climbs to 8), but within ~1,000 steps, and a ridge filter fits better almost throughout.
    The accurate contrast is a slow PCR staircase (AdamW) vs a compressed, ridge-like progression
    (Muon).
  - This does not depend on how checkpoints are weighted. Evaluations are 4× denser before step
    2,000, but medians taken uniformly in steps agree within 0.07 and never change a run's side
    of 1.
  - On isotropic inputs both optimizers are ridge-like (1.7–1.9), as expected when every axis
    is equivalent.
  - **"All end near least squares" holds in distribution, but not on every axis within this
    budget.** Two of six AdamW 1/i² runs finish the decay (10k steps) with implied-weight gain
    ≈ 0 on the weakest axis. One aligned seed also has gain ≈ 0 on axis 7, so it is effectively
    a PCR(6) solution. Every Muon run has gains 0.95–1.01 on all eight axes.
* **H3 — confirmed at matched in-distribution error: 1.5–2.5× lower shifted-query error for
  Muon.** Each run's trajectory is interpolated (log-linear, checkpoints 500 steps apart) to the
  step where its in-distribution query error first reaches each level. This is the proposal's
  "similar ordinary error" comparison.

  | shifted-query error when in-distribution error reaches | 0.03 | 0.01 | 0.005 |
  |---|---|---|---|
  | 1/i² aligned — AdamW | 0.47 [0.43–0.52] (2 seeds) | 0.22 [0.217–0.222] (2) | 0.15 (1) |
  | 1/i² aligned — Muon | 0.30 [0.26–0.33] | 0.10 [0.102–0.104] | 0.062 [0.059–0.063] |
  | 1/i² rotated — AdamW | 0.46 [0.37–0.55] (3) | 0.20 [0.13–0.26] (2) | 0.15 [0.10–0.20] (2) |
  | 1/i² rotated — Muon | 0.31 [0.30–0.32] | 0.12 [0.117–0.131] | 0.060 [0.0595–0.0598] |

  - Muon is lower at every level: 1.5–1.6× at 0.03, 1.6–2.2× at 0.01, 2.5× at 0.005. Within
    this budget the gap widens as training proceeds instead of closing, unlike the testbed,
    where all optimizers reach the same solution.
  - Aligned, the seed ranges never overlap. Rotated, the best AdamW seed equals Muon at 0.01
    (0.127 vs 0.117–0.131) and is 1.7× worse at 0.005.
  - AdamW seeds that never reach a level drop out of that column, so the averages flatter AdamW.
  - After the decay (not matched): 0.040 vs 0.34 aligned and 0.031 vs 0.27 rotated. AdamW's
    spread (0.09–0.66) is set by whether the seed learned the weakest axis.
  - The mechanism matches H4. At equal ordinary error, the PCR-like path has put less weight on
    the weak axes, and the inverted-spectrum query loads exactly those axes.
* **Aligned vs rotated — the null result for AdamW replicates.** It converged at 5,250
  [4,900–5,600] (+1 never) aligned vs 6,200 [4,900–6,900] rotated. Error@32 at the end of the
  constant-LR phase is 0.021 [0.009–0.042] vs 0.023 [0.010–0.039], and the PCR/ridge ratio 0.76
  vs 0.80. All differences are within seed noise. Muon converged at 867 vs 888.
* **Garg et al.'s out-of-distribution suite, on the decayed isotropic-trained models (3 seeds
  each; figure below).**
  - In distribution, both match least squares for k ≥ 16 (error ≤ 0.025 AdamW, ≤ 0.006 Muon).
    Near k = d = 8, where least squares becomes exact, Muon is closer (0.11 vs 0.15 at k = 8).
  - Noisy labels (σ = 1): both track the Bayes ridge estimator, as Garg et al. found. At k = 32
    they reach 0.16–0.19 vs 0.15. At k = 8 they reach 0.48–0.63 vs 0.40, while least squares
    blows up to 744. Here AdamW is slightly closer than Muon near k = d (0.48–0.56 vs 0.58–0.63).
  - Orthogonal queries: both ≈ 1.0, as expected. Half-dimensional subspace and inputs ×2: Muon is
    slightly lower (0.002 vs 0.011 and 0.023 vs 0.038 at k = 32).
  - **Skewed covariance (Garg's failure case):**
    - All 3 Muon models keep improving with k (error@32 0.023–0.033).
    - Only 1 of 3 AdamW models does (0.029). One gets worse with k (0.19 at k = 7 → 0.37 at
      k = 32), and one is erratic (0.93 at k = 16, 0.16 at k = 32).
    - Counting the pilot and Garg-recipe models too: 5/5 Muon-trained models are robust vs 2/6
      Adam-trained ones.
    - This is suggestive, not established: n is small, and the Adam models span three LRs and
      two token layouts. The robust AdamW seed is also the one closest to least squares in
      distribution at k = 8 (0.113 vs 0.168 and 0.179).
* **Head alignment (exploratory) — the pilot's layer-specific claims do not replicate; one
  pattern does.**
  - Across fresh seeds, the AdamW layer-3 concentration on the top axes that the pilot reported
    varies from seed to seed. On isotropic inputs only AdamW's layer 2 is fully diagonal in
    every seed; Muon's is diagonal in only some heads.
  - What holds in every run is layer 2 under 1/i². Muon's heads cover all eight axes, with a
    share that falls steadily with the eigenvalue (0.20–0.36 on axis 1, 0.04–0.05 on axis 8;
    5/5 runs). AdamW's layer-2 share on axis 8 is 0.00 in 6/6 runs.
  - The four AdamW runs that do learn the weakest axis carry it in layer 3 or 4 (axis-8 share up
    to 0.16–0.27). The two whose weakest-axis gain ends ≈ 0 have zero share on that axis in
    every layer. So the head-level picture agrees with the implied-weight gains
    (`figures/confirm/heads.txt`).

![Confirmation learning curves (mean and seed range)](figures/confirm/curves_err32.png)
![Per-axis learning times, seed 10](figures/confirm/axis_times.png)
![Best-fitting estimator along training, seed 10](figures/confirm/algo.png)
![Per-axis implied-weight gains, seed 10](figures/confirm/implied_gain.png)
![Query shift at matched in-distribution error, all seeds](figures/confirm/qshift_traj.png)
![Garg et al. suite on the confirmation models (isotropic-trained, after decay)](figures/confirm/garg_eval_confirm.png)
