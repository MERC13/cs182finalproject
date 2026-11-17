## The theory testbed (one-layer linear attention) supports H1, H2 and H4

Same data as the proposal (d = 8, N = 32, noiseless, the five covariances). Model: Zhang et al.'s
8-head linear attention, either **separate rank-1 K and Q per head** (the setting of their
staircase) or **merged K·Q**, small init. Each optimizer got its own 5-point LR sweep per
condition (selection: area under the loss curve), then 3 seeds at the selected LR, 4,000 steps,
batch 1,024. The loss is the exact population loss; the optimum is one-step preconditioned GD
(trΛ-shrunk), so "converged" means within 5% of that optimum.

**Steps to converge (excess loss < 5%), separate K/Q, median of 3 seeds**

| | iso | 1/i aligned | 1/i rotated | 1/i² aligned | 1/i² rotated |
|---|---|---|---|---|---|
| GD | 460 | 1,590 | 1,350 | > 4,000 | > 4,000 |
| Adam | 90 | 110 | 240 | 60 | 390 |
| SignGD | 110 | 140 | 350 | 100 | 1,000 |
| Muon (NS5) | 70 | 70 | 70 | 70 | 60 |
| Muon (exact polar) | 60 | 60 | 60 | 50 | 60 |

![per-axis gains](figures/linattn/axes_sep.png)
![per-axis learning times](figures/linattn/axis_times_sep.png)

**Robustness check (added in verification).** I reran every optimizer on the *exact* population
gradient (no sampling noise), over an 8-point LR grid, at two initialization scales (1e-3 and
1e-2). The table gives steps to converge at each optimizer's best LR (both inits agree within a
factor of 1.5):

| λ ∝ 1/i², aligned → rotated | separate K, Q | merged K·Q |
|---|---|---|
| GD | > 3,000 → > 3,000 | 330 → 330 |
| Adam | 20 → 170–180 | 20 → 70 |
| SignGD | 30 → ≈1,450 | 30–40 → 610–630 |
| Muon (NS5) | 10–20 → 30 | 10–20 → 20 |
| Muon (exact polar) | 10 → 10 | 10–20 → 10 |

So the orderings below do not depend on minibatch noise, the LR picked, or the init scale.
Muon and GD are rotation-invariant, Adam and SignGD are not, and GD is slowest. With exact
gradients the normalised optimizers converge in tens of steps at the top of the grid, so only
the *relative* comparisons are meaningful here.

* **H1 (Muon earlier): supported.** Muon is the fastest in every anisotropic condition.
  On isotropic inputs Adam is about as fast (90 vs 70 steps).
* **H2 (imbalance slows SGD more than Muon): supported, strongly.** GD's time to learn an axis
  scales roughly as λ⁻², matching Zhang et al.'s plateau law. Under λ ∝ 1/i², GD has not learned
  axes 4–8 after 4,000 steps. Muon's convergence time is flat across all five covariances, the
  condition-number-free behaviour Ma et al. prove.
* **Alignment (the proposal's R manipulation): Adam and SignGD depend on it; GD and Muon do not.**
  Rotating the covariance slows Adam 2.2× (1/i) to 6.5× (1/i²), and SignGD 2.5–10×. GD and Muon
  are unchanged, as rotation equivariance requires. This is the testbed analogue of Xie et al.
  When aligned, Adam is nearly as fast as Muon, as Ma et al.'s SignGD = Muon identity predicts.
* **H4 (SGD → PCR staircase, Muon → ridge-like): supported, but only for rank-1 separate heads.**
  - GD learns one principal axis at a time, in eigenvalue order. Its gain profile along the
    learning path fits a PCR step better than a ridge curve in every anisotropic condition
    (path-averaged residual 0.01–0.20 vs 0.42–0.77).
  - Exact-polar Muon learns all axes nearly together and is ridge-shaped (0.44 vs 0.10 at 1/i²
    aligned). NS5-Muon is in between at 1/i² (0.25 vs 0.31). Its quintic iteration
    under-amplifies the smallest gradient singular values, so the weakest axis takes 160 steps
    instead of 90.
  - Adam and SignGD are PCR-like under 1/i² even when aligned, i.e. closer to SGD than to Muon,
    which H4 does not anticipate.
  - **With merged K·Q the staircase disappears for every optimizer, GD included.** GD's profile
    becomes ridge-like (0.34 vs 0.20 at 1/i aligned), with one abrupt drop, as Zhang et al.
    predict. So the PCR path is a property of the *parametrization* as much as of the optimizer.
    That matters for a GPT-2 whose heads have dimension 32 ≥ d.

    ![merged K·Q testbed](figures/linattn/axes_merged.png)
* **H3 (smaller query-shift error): supported along the path, but not at convergence.**
  - *At equal steps:* after 4,000 steps under λ ∝ 1/i², the inverted-spectrum query loss is 0.96
    for GD and 0.74 for rotated Adam, against 0.65 for Muon.
  - *At matched in-distribution loss* (the proposal's "similar ordinary error" test), Muon is
    still better: when excess loss is 0.1, the query-shift loss is 0.50 (NS5) and 0.44 (exact)
    vs 0.60 for GD and 0.57 for Adam (1/i aligned, separate K/Q). Under 1/i² aligned it is
    0.92 / 0.88 vs 0.97 / 0.97. The reason is that at equal ordinary loss a ridge-like path has
    already partly used the weak axes, while a PCR path has dropped them.
  - *At convergence* every optimizer reaches the same solution (≈0.65 under 1/i²; the optimum
    itself is not shift-robust), so the advantage is transient.

  ![testbed query shift at matched loss](figures/linattn/qshift_traj_sep.png)
* **Caveat:** with a constant LR, Muon's normalised step never shrinks. It ends with a slightly
  higher noise floor than Adam (0.226 vs 0.222 on iso). Ma et al.'s rate assumes a decaying step
  size, so the full study should use a schedule.
