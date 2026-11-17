## Appendix — the pilot's original GPT-2 results (superseded by verification step 3)

*Kept for transparency. These runs used learning rates tuned only on isotropic inputs and mostly one seed, evaluated on the selection prompts. Verification step 3 replaces them.*

LRs tuned on isotropic inputs (Muon 0.005 with aux AdamW 1e-3; AdamW 3e-4; SGD 0.1 with
gradient clipping 1.0, its best stable setting), then reused on every covariance. One seed per
cell unless noted. "Converged" = error averaged over k = 1..32 within 0.05 of least squares.
"Axis learned" = relative error on that axis's probe tasks below 0.5.

Seed-1 values, where run, are in brackets. SGD was run only on isotropic inputs.

| covariance | optimizer | leaves plateau | half-way | converged | weakest axis learned | axis-8/axis-1 time | in-dist. error @32 | shifted-query error (t=1) | PCR/ridge fit ratio |
|---|---|---|---|---|---|---|---|---|---|
| isotropic | SGD | – | – | – | – | – | 1.036 | 1.002 | – |
| isotropic | AdamW | 1,900 | 2,500 | 5,600 | 2,500 | 1.0 | 0.014 | 0.008 | 1.71 |
| isotropic | Muon | 400 (400) | 600 (500) | 1,100 (1,200) | 500 (500) | 0.8 (0.8) | 0.009 (0.009) | 0.005 (0.006) | 1.69 (1.66) |
| 1/i aligned | AdamW | 400 | 1,100 | 5,600 | 2,600 | 5.2 | 0.014 | 0.027 | 1.75 |
| 1/i aligned | Muon | 300 | 400 | 1,000 | 600 | 2.0 | 0.005 | 0.008 | 1.55 |
| 1/i² aligned | AdamW | 200 (300) | 300 (300) | 6,700 (6,600) | 8,500 (7,800) | 42.5 (26.0) | 0.009 (0.013) | 0.174 (0.182) | 0.88 (0.88) |
| 1/i² aligned | Muon | 200 (300) | 300 (300) | 900 (900) | 900 (900) | 3.0 (3.0) | 0.006 (0.004) | 0.065 (0.033) | 1.21 (1.23) |
| 1/i² aligned | Muon + read-in | 200 | 300 | 1,200 | 1,800 | 6.0 | 0.004 | 0.045 | 1.11 |
| 1/i² rotated | AdamW | 200 | 300 | 6,300 | 7,400 | 37.0 | 0.009 | 0.135 | 0.84 |
| 1/i² rotated | Muon | 200 | 300 | 1,000 | 1,000 | 3.3 | 0.004 | 0.043 | 1.24 |
| 1/i² rotated | Muon + read-in | 300 | 300 | 1,400 | 1,800 | 6.0 | 0.004 | 0.045 | 0.88 |

![learning curves](figures/grid/curves_err32.png)
![per-axis implied-weight gains](figures/grid/implied_gain.png)
![per-axis learning times](figures/grid/axis_times.png)
![best-fitting estimator along training](figures/grid/algo.png)

* **H1 (Muon reaches ICL earlier): supported, large effect.**
  - On isotropic inputs Muon leaves the predict-zero plateau at step ~400 and converges by
    ~1,100. AdamW does so at ~1,900 and ~5,600.
  - SGD + momentum never left the plateau in 10,000 steps, nor at lr 0.01 or 0.03 in 4,000.
  - Across all four covariances Muon converges 5–7× sooner than AdamW (900–1,100 vs
    5,600–6,700 steps).
  - The gain comes from Muon's updates to the hidden matrices. With its auxiliary AdamW LR
    lowered to AdamW's own 3e-4, Muon still escapes by ~500 steps (error 0.21 at step 1,000 vs
    0.17).
  - Caveat: "first escape" is earlier on skewed spectra for both optimizers (step 200 under
    1/i²), because one dominant axis is easy to learn. Compare convergence and weakest-axis
    times, not first escape.
* **H2 (imbalance slows SGD more than Muon): supported with AdamW in SGD's place.** SGD never
  leaves the plateau even on isotropic inputs, so it cannot be scored here. AdamW is clearly
  slowed by imbalance and Muon barely is: weakest-axis time 2,500 → 2,600 → 8,500 steps for
  AdamW (isotropic → 1/i → 1/i²; 7,800 with the second seed) vs 500 → 600 → 900 for Muon
  (both seeds). The ratio of last-axis to first-axis time grows 1 → 5 → 26–42 for AdamW and only
  1 → 2 → 3 for Muon.
* **Aligned vs rotated: no effect in the GPT-2.** AdamW converges at 6,700 (aligned) vs 6,300
  (rotated) steps, learns the weakest axis at 8,500 vs 7,400, and the curves overlap. In the
  testbed, rotation slowed Adam 6.5×. Only the read-in sees the input basis, and AdamW's
  hidden-layer coordinates are random relative to it. As designed, R is a near-null
  manipulation for this architecture.
* **H4 (SGD → PCR on a growing set, Muon → ridge-like): the staircase is real, but it belongs to
  AdamW.**
  - Under 1/i², AdamW learns the principal axes one at a time, in eigenvalue order (steps 200,
    600, 1,500, 2,500, 4,500, 5,500, 7,500, 8,500). The best-fitting PCR rank climbs 1 → 2 → … → 8 at roughly one rank per 1,200 steps (rank 8 at step 9,000), aligned and rotated alike, and PCR fits its predictions better than any ridge λ from step ~1,000 until rank ≈ 7 (median distance ratio 0.88 aligned, 0.84 rotated).
  - Muon learns all eight axes within ~3× of each other (300–900 steps) and is fit better by
    ridge (ratio 1.21 / 1.24). This matches the testbed and Ma et al.'s λ-proportional filter.
  - On isotropic and 1/i inputs both optimizers are ridge-like (ratios 1.5–1.75).
  - SGD's predicted staircase cannot appear here, because SGD never forms the first
    (copy-x-into-the-y-token) circuit. Head alignment shows its key-side input map stays at
    R² ≈ 0.01 in every layer.
* **H3 (Muon more robust to queries on weak axes): supported, including at matched ordinary
  error.**
  - At the end of training on 1/i², in-distribution error is ≈0.004–0.013 for both, but the
    inverted-spectrum query error is 0.174 / 0.182 (AdamW, two seeds) vs 0.065 / 0.033 (Muon)
    aligned, and 0.135 vs 0.043 rotated. AdamW was trained twice as long (10k vs 5k steps).
  - At matched in-distribution error (≈0.007–0.009) it is 0.23 vs 0.12 aligned and 0.135 vs
    0.117 rotated. Under 1/i the two trajectories overlap, so the effect grows with imbalance
    (figure below).
  - AdamW's final implied-weight gain on the weakest axis is 0.90 vs 0.99 for Muon. The
    training loss barely penalises this (axis 8 carries 1% of the variance), but shifted
    queries expose it.

  ![query shift at matched in-distribution error](figures/grid/qshift_traj.png)
* **Muon with the read-in included (the exactly rotation-invariant arm).** Aligned and rotated
  runs are indistinguishable (weakest axis learned at step 1,800 in both, shifted-query error
  0.045 in both), as equivariance requires. Moving the read-in from AdamW to Muon slows the weakest
  axes about 2× relative to standard Muon (1,800 vs 900–1,000 steps), so the AdamW-updated read-in
  helps the weak directions. Even so, it is 4–5× faster on them than AdamW.
* **Replication.** Second seeds closely reproduce the first:
  - Muon on isotropic inputs escapes at 400 and converges at 1,200 vs 1,100.
  - Muon on 1/i² aligned learns the axes at the same steps (300 … 900).
  - AdamW on 1/i² aligned repeats the staircase: axes at 300, 500, 1,300, 2,700, 4,400, 5,200, 6,600 and 7,800 (seed 0: 200 … 8,500). It converges at 6,600 vs 6,700, ends at shifted-query error 0.182 vs 0.174, and has the same PCR/ridge ratio of 0.88.
* **Head alignment (exploratory).** On isotropic inputs, both optimizers develop layer-2 heads
  that are almost exactly diagonal in the whitened principal basis (share 0.97–0.99), spread
  evenly over the eight axes: an x_qᵀΣ⁻¹x_k-style comparison after a layer-1 copy step. Under
  1/i², both concentrate on the top axes; AdamW more so (rotated, layer 3: 0.48 / 0.25 on axes
  1–2 and ≈0 on axes 7–8, vs 0.30 / 0.16 … 0.07 / 0.07 for Muon).
