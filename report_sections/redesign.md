## A better experiment, given the negative results

Three of the proposal's manipulations came back null or untestable in the GPT-2:
- SGD never trains it.
- Rotating the input covariance does nothing for AdamW.
- Muon is not more transient than AdamW at matched training loss.

This section asks why, using a literature review (2023–2026; papers read in full where cited as such) and pilot runs. It then recommends a redesign in which each of these becomes a measured boundary condition instead of a dead end. Pilot code and logs:
- `scripts/noise_sweep_linattn.py`, `scripts/ladder_sweep.py`, `scripts/analyze_noise.py`, `icl/lintf.py`;
- `runs/linattn/noise_sweep*.json`, `runs/noise/`, `runs/concat/`, `runs/ladder/`, `runs/zipf/`.
Figures are in `figures/noise/`.

### What the negative results have in common

**1. SGD: the GPT-2 is outside the regime where SGD is known to train.**
- No paper the review found trains Garg et al.'s GPT-2 on in-context regression with SGD. The models trained with GD or SGD in the ICL theory literature are linear or one-layer attention models, mostly on concatenated (x, y) tokens (e.g. Zhang, Frei & Bartlett 2023; Huang, Cheng & Liang 2023; Ahn et al. 2023).
- Our measured causes match the literature:
  - Block heterogeneity (Zhang et al. 2024, arXiv 2402.16788).
  - Plateau gradients that are nearly rank-1 along the read-out direction. A scalar read-out sends one direction back into the residual stream at every token.
  - Interleaved x and y tokens need a two-layer copy circuit before any regression can happen (von Oswald et al. 2023, arXiv 2212.07677).
- Pilot: concatenated tokens [x_i, y_i] make AdamW about 10× faster (error 0.24 by step 400 on isotropic inputs). SGD still stays at error 1.00 for 3,000 steps at lr 0.1 and 0.3.
- **Adding a larger init fixes it.** Every hidden matrix gets std 1/√128 instead of GPT-2's 0.02 and 0.02/√8, which shortens the chain of tiny matrices the gradient passes through (Noci et al. 2022, arXiv 2206.03126). SGD (lr 0.1, momentum 0.9, clip 1) then reaches error 0.25 by step 600 and 0.16 by step 3,000 on isotropic inputs. AdamW with the same init gets 0.15.
- **Both changes are needed.** The larger init alone, with separate x and y tokens, leaves SGD at error 0.96 after 3,000 steps.

**2. Rotation: rotating x is the one kind of rotation Adam tolerates in transformers.**
- x → Rx changes the read-in W_in to W_in·diag(R, 1). That rotates the weights *leading to each read-in neuron* among themselves, which Zhang, Maes et al. (2024, arXiv 2410.19964, full text) call an *output-wise* rotation. In their words: "Performance is preserved under output-wise rotations but progressively worsens with input-wise, layer-wise, and global rotations" (GPT-2: 16% slower under a global rotation; ViT: 96%).
- Adam-mini (arXiv 2406.16793) makes the same point another way: one learning rate per head or output neuron matches AdamW. So the within-neuron adaptivity an input rotation could touch is not what Adam relies on in a transformer.
- Pilot: noise does not explain the GPT-2 null. At batch 256 (4× ours), AdamW learns axes 1–5 at 125, 250, 500, 1,000 and – steps aligned, against 125, 275, 575, 975 and 1,500 rotated. Its spectral exponent over axes 1–4 is 0.73 vs 0.74.
- In the one-layer testbed, where the rotation does reach the attention weights directly, Adam's penalty *grows* with batch size. This is the sign-descent limit (Kunstner et al. 2023, arXiv 2304.13960; Balles & Hennig 2018).

**3. Simplicity bias: gradient noise is the dial, and batch 64 was already near its low-noise end.**
- Spectral descent learns all directions at once only without noise. Small-batch noise restores sequential learning, because when the gradient is small relative to the noise, the expected orthogonalised update becomes proportional to the gradient (Jacobs & Burkholz 2026, arXiv 2610.16868, full text).
- Even without noise, spectral flow on linear attention spreads learning times by (λ_j/λ̂_j)^(1/3) (Dragutinović et al. 2026, arXiv 2603.00742, Thm C.3). For our spectrum that is about 2.6–4× at N = 32, i.e. a floor of β ≈ 0.25–0.33.
- Pilot results for β, the spectral learning exponent (time to learn eigen-direction j ∝ λ_j^(−β)):

| Muon, λ ∝ 1/i² | batch 16 | 64 | 256 | 1,024 |
|---|---|---|---|---|
| testbed, Newton–Schulz, momentum 0.95 | 0.90 | 0.68 | 0.56 | 0.37 |
| testbed, exact polar, momentum 0.95 | 0.69 | 0.64 | 0.41 | 0.25 |
| testbed, Newton–Schulz, no momentum | 0.84 | 0.78 | 0.66 | 0.43 |
| GPT-2, Newton–Schulz, momentum 0.95 | 0.45 | 0.33–0.38 (3 seeds) | 0.25 | – |

*Testbed: 3 seeds at a per-cell tuned LR. GPT-2 at batch 16 and 256: one seed each. For reference, AdamW in the GPT-2 has β ≈ 0.8–1.1, and GD in the testbed β ≈ 1.3–1.7 on the axes it learns.*

![Spectral learning exponent and convergence time vs batch size, one-layer testbed](figures/noise/testbed_noise.png)

*One-layer testbed, λ ∝ 1/i², 3 seeds per cell at a tuned LR. Left: β for each optimizer, aligned (solid) and rotated (dashed). Right: steps to converge. Adam's and SignGD's rotation penalty (dashed above solid) grows with batch size; Muon is rotation-invariant and becomes more sequential as the batch shrinks.*

- Noise raises β in every Muon variant. Momentum and exact orthogonalisation lower it.
- The GPT-2 at batch 64 (32 predictions per prompt, momentum 0.95) already sits close to the noise-free floor. So the transience test ran where Muon's simplicity bias should be *weakest*, and Muon was still not more transient. Noise cannot explain that null away.

**4. The model ladder: where SGD trains and where Adam's alignment advantage survives.**
- Multi-layer linear transformers on raw concatenated tokens (`icl/lintf.py`; separate per-head Q, K, V, O, no softmax, MLP or LayerNorm).
- Setting: λ ∝ 1/i², batch 256, 2,000 steps, a 3-point LR sweep per cell and 2 seeds.

| model | SGD: steps to loss 0.25 | SGD β (aligned / rotated) | Adam β | Muon β |
|---|---|---|---|---|
| 1 layer, raw tokens | 400 | 0.79 / 0.77 | **0.37 / 0.68** | 0.29 / 0.37 |
| 2 layers, raw tokens | 200 | 0.77 / 0.84 | 0.67 / 0.69 | 0.53 / 0.60 |
| 4 layers, raw tokens | 200 | 0.80 / 0.84 | 0.75 / 0.80 | 0.44 / 0.43 |
| 2 layers + learned 32-wide read-in | **never** (loss 0.45 / 1.07) | – | 0.68 / 0.89 | 0.55 / 0.58 |

- **SGD trains multi-layer linear transformers on raw tokens at every depth**, as Ahn et al. (2023, arXiv 2310.01082) report. It fails as soon as a learned read-in into a wider stream is added. The read-in and width are part of SGD's GPT-2 problem.
- **Adam's alignment advantage is clear only in the one-layer raw-token model**: β 0.37 aligned vs 0.68 rotated, and loss 0.25 reached at step 100 vs 250. At depth 2 and 4, and with a read-in, it is within noise. The GPT-2 null is the end point of this trend.
- **Muon has the lowest β at every depth.**
- *Caveats.* This was one quick pass.
  - Adam at depths 2 and 4, and SGD at depth 4 (aligned), diverged late in training at the selected LR. Their times and β come from before the divergence.
  - SGD and Muon are rotation-equivariant, so their aligned-vs-rotated differences should be zero. Those differences (up to 2× in timing, 0.1 in β) are this pilot's noise floor.
  - A full version needs ≥ 4 seeds, finer logging and an LR decay.

**5. A Zipfian task pool (cheap version) shows only a small, early effect.**
- Setting: 64 training tasks with frequencies ∝ 1/k (Chan et al. 2022), label noise 0.5, one seed each.
- Memorisation is scored separately on the most frequent eighth of the tasks and the least frequent half, at matched training loss:

| matched training loss (m) | AdamW: rare − frequent | Muon: rare − frequent | ICL score on unseen tasks (AdamW / Muon) |
|---|---|---|---|
| 0 | −2.63 | −2.18 | 0.43 / 0.53 |
| 0.25 | −1.95 | −1.72 | 0.60 / 0.67 |
| 0.5 | −1.42 | −1.41 | 0.71 / 0.73 |

- Muon is slightly more balanced across task frequencies early on, the direction Wang et al. (2025, arXiv 2509.26030) predict. It also does more ICL, as in the uniform-pool test. The gap closes by m = 0.5.
- In this version the task enters only through the context, so nothing reaches Adam's per-row adaptivity. That adaptivity is the mechanism by which heavy-tailed data separates Adam from GD (Kunstner et al. 2024, arXiv 2402.19449).
- If this direction is pursued, give each task an identity token whose embedding row the model learns. The rare tasks then live in rarely updated rows. A fixed random rotation of that table is invisible to SGD and Muon but not to Adam, which also checks the code.

**6. The redesigned GPT-2: all three optimizers train, and their order is the one the proposal predicted.**
- Setting: concatenated tokens and hidden init 1/√128; λ ∝ 1/i² aligned, batch 64, the study's tuned LRs (SGD 0.1 with clipping); seeds 10 and 11.

| optimizer | steps at which axes 1–8 are learned (seed 10; seed 11) | β | weakest ÷ strongest | converged | final error |
|---|---|---|---|---|---|
| SGD | 25, 50, 150, 275, 425, 850, 1,600, 2,100; 25, 100, 150, 225, 350, 1,450, 1,975, 2,500 | **1.11, 1.11** | 84×, 100× | 2,300, 2,500 | 0.071, 0.077 |
| AdamW | 50, 75, 125, 200, 325, 600, 950, 1,250; 50, 75, 175, 250, 400, 825, 1,475, 2,000 | **0.80, 0.91** | 25×, 40× | 1,425, 2,000 | 0.066, 0.072 |
| Muon | 50, 75, 100, 150, 200, 250, 325, 400; 50, 75, 100, 150, 200, 325, 450, 600 | **0.51, 0.60** | 8×, 12× | 450, 575 | 0.065, 0.064 |

- SGD is the most sequential and Muon the least, with AdamW between, in both seeds and with no overlap.
- All three end at similar error, and everything learns faster than in the original layout: AdamW converges at 1,425 vs ~5,000 steps.
- One caveat. The PCR-vs-ridge path ratio, which separated AdamW (PCR-like) from Muon (ridge-like) in the original model, reads 1.28–1.38 (ridge-like) for all three here. The estimator-shape part of H4 needs re-checking in this model.

### Recommended experiment

**Question.** How do an optimizer's geometry and the gradient noise set the order in which a transformer learns the directions of an in-context regression task? And does that order predict out-of-distribution behaviour?
- The geometries are Euclidean (SGD), coordinate-wise (AdamW) and spectral (Muon).
- This keeps the proposal's core (H1, H2, H4: who learns which directions, when) and fixes each failure:
  - SGD now trains.
  - The factor that replaces rotation, batch size, demonstrably moves the optimizers.
  - The simplicity-bias question becomes a dose-response measurement instead of a transience test.

**Model.** The study's GPT-2 (4 layers, 4 heads, width 128) with concatenated (x_i, y_i) context tokens, which keep a prediction at every prefix length, and hidden matrices initialised at std 1/√width. Optionally add the one-layer linear-attention testbed and 2–4-layer linear transformers as bridges to theory.

**Factors.**

| factor | levels | why |
|---|---|---|
| optimizer | SGD (momentum 0.9, clip 1), AdamW, Muon (Newton–Schulz, momentum 0.95). Controls: normalised SGD; Muon with exact polar or no momentum | the proposal's three geometries; normalised SGD is Muon without the orthogonalisation; exact polar is Muon as the theory has it; no momentum removes momentum's noise averaging |
| batch size (gradient noise) | 16, 64, 256 (1,024 on a GPU) | moves Muon's β from 0.45 to 0.25 in the original GPT-2 and from 0.90 to 0.37 in the testbed |
| input spectrum | λ ∝ i^−p, p = 0, 1, 2 | the range of λ that β is fitted over; p = 0 is the control (all directions equal) |
| seeds | ≥ 3 per cell, 5 for AdamW | AdamW's weak-axis times vary most across seeds |

**Readouts.**
1. β: the slope of log(time to learn eigen-direction j) against log(1/λ_j), from probe prompts (`run_metrics` already computes the axis times).
2. The weakest ÷ strongest learning-time ratio, and the final per-axis gains.
3. At matched in-distribution error: error on queries that load the weak directions (the proposal's H3), and Garg's skewed-covariance test.

**Hypotheses, with what the pilots already show.**
- **R1, geometry orders β.** At every batch size, β_SGD > β_AdamW > β_Muon.
  - Redesigned-GPT-2 pilot, two seeds: 1.11 > 0.80–0.91 > 0.51–0.60.
  - Theory: GD's plateau law gives time ∝ λ⁻² in linear attention (1.3–2 in our testbed). Noise-free spectral flow has a floor of β ≈ 0.25–0.33 for this spectrum.
- **R2, noise dial.** β_Muon falls toward that floor as the batch grows; β_SGD barely moves; β_AdamW lies between.
  - Original-GPT-2 pilot: Muon 0.45 → 0.33–0.38 → 0.25 for batch 16 → 64 → 256 (one seed at 16 and 256). AdamW ≈ 0.9 → 0.73 over axes 1–4 for batch 64 → 256.
  - Still to be run in the redesigned model.
- **R3, does the order matter?** At matched in-distribution error, weak-direction query error and Garg's skewed-covariance failure should track β: worst for SGD, best for Muon, and worse for Muon at small batch than at large batch. This is the testable form of "simplicity bias costs or buys something".
- **R4, negative control.** Input rotation should change nothing in the GPT-2 (pilot: AdamW null at batch 64 and 256). Keep one rotated cell per optimizer as a check, not as a hypothesis.

**Drop or demote:** the aligned-vs-rotated arm as a hypothesis, and transience as the main question. Transience was null in the regime where Muon's simplicity bias is weakest.

**Cost.**
- On CPU a GPT-2 run takes about 10–30 minutes at batch 64; batch 256 is about 4× slower per step.
- The core grid is 3 optimizers × 3 batch sizes × 3 spectra × 3 seeds = 81 runs. That is roughly a day on a 4-core machine, or a few hours on one GPU.
- The linear bridge models take minutes per run.

**Second-choice designs, ranked.**
1. *Adam's basis dependence, measured directly.* Rotate AdamW's parameter coordinates at increasing scope: output-wise, input-wise, layer-wise, global (Zhang, Maes et al. 2024). Compare with Adam-mini.
   - Prediction: output-wise ≈ null, which is what input rotation amounts to; global ≫ input-wise.
   - About 8 CPU-hours.
2. *Zipfian task identities through a learned embedding table* (point 5 above). This reaches Adam's per-row adaptivity, and Muon is predicted to be more balanced across task frequencies. About 15 CPU-hours.
3. *Transience at the two ends of the noise dial* (batch 16 vs ≥ 256) with finite task pools. This is a small-effect test of the original hypothesis; our data predict no difference at batch ≥ 64.

**Novelty.** The literature review found none of the following published:
- a study of Muon in ICL beyond one layer;
- a test of how batch size changes the order in which optimizers learn ICL directions;
- a clean test of Adam's rotation penalty against batch size.

The recommended study would supply all three.
