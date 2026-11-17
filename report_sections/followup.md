## Follow-up: four questions after verification

### 1. Why can't SGD train this GPT-2?

SGD stays on the initial plateau, where the model predicts ≈ 0, for two separate measured reasons.
Both are fixed by Muon's orthogonalised update, and the first by AdamW's per-coordinate one.

**(a) One learning rate for very different blocks.** Per-block gradients at the plateau (mean of 16
batches, `scripts/grad_blocks.py`):

| block | share of the squared gradient norm | gradient per parameter (rms) | change per step: SGD lr 0.1, momentum 0.9 (steady state, before clipping) | AdamW 5e-4 | Muon 0.005 |
|---|---|---|---|---|---|
| read-out | 63% | 3.0e-1 | **576%** of its norm | 1.0% | 1.9% |
| attention output / MLP projection | 6% / 16% | 4.0e-3 / 3.4e-3 | 56% / 48% | 7.1% / 7.1% | 6.3% / 3.1% |
| query / key | 0.0007% each | 4.4e-5 / 4.3e-5 | **0.22% / 0.21%** | 2.5% / 2.5% | 2.2% / 2.2% |

- SGD's usable learning rate is capped by the read-out. That is why lr 0.1 needs gradient
  clipping and lr 1.0 is unstable. At that cap the query/key weights move 2,600× more slowly than
  the read-out.
- AdamW and Muon move every block at 0.3–7% per step. This is Zhang et al.'s (2024) "block
  heterogeneity" explanation of why transformers need Adam, measured directly here.
- Moving the read-out, embeddings and norms to AdamW does not rescue it: SGD on the hidden
  matrices alone is still at error 1.000 after 3k steps. That was run at lr 1 and 10 with Muon's
  averaged momentum, roughly heavy-ball lr 0.1 and 1.

**(b) Within each matrix, the gradient is essentially one direction** (`scripts/grad_spectrum.py`).
- On the plateau, the top singular direction of every hidden matrix's gradient carries 85–100% of
  its squared norm (effective rank 1.1–2.8; single minibatches look the same).
- In the attention-output and MLP-projection matrices that direction is 99–100% aligned with the
  read-out, i.e. it only rescales the prediction.
- *Normalised SGD* (`--muon_orth fro`) takes exactly Muon's per-matrix step size, in the
  gradient's own direction. It is still at error 1.000 after 3k steps at lr 0.005 and 0.01, while
  Muon at lr 0.005 escapes at ~430 steps.
- So balancing step sizes across blocks is not enough. Muon works because orthogonalisation gives
  all 128 directions of each matrix the same step, including the weak ones that build the
  in-context circuit, instead of spending the step on the output-scale direction.
- Earlier checks still stand: SGD also fails at d = 2, with Garg's curriculum, and at lr 0.01–1.0
  with and without clipping.

Consequence for the proposal: the SGD arm cannot be run in this model. Normalised SGD is a useful
extra ablation ("Muon without orthogonalisation"), but it does not learn either. The clean SGD
comparison lives in the linear-attention testbed, where GD does learn.

### 2. Is "Muon is better for skewed covariance" the key takeaway?

Only in a narrower form:

> With skewed input covariance, AdamW learns in-context regression one principal direction at a
> time, so the weak directions arrive late (7× later than with Muon, and in 2 of 6 runs not at all
> within 10k steps). Muon learns them nearly together, so it converges 6–7× sooner. At equal
> in-distribution error, it is 1.5–2.5× more accurate on queries that load the weak directions.

- *Not proven.* This is evidence from one 0.8M-parameter model, d = 8, noiseless data, 3 seeds and
  ≤ 10k CPU steps. The only proofs are for one-layer linear models (Ma et al.).
- *"Better" means faster, per step, and more balanced across directions.* Both optimizers end
  near least squares in distribution. Wall-clock and compute were not compared. Robustness to a
  skewed covariance at *test* time (Garg's test) is only suggestive (5/5 vs 2/6 models).
- *Muon's advantage comes from a setting with nothing to overfit.* Every input direction carries
  real signal, tasks are fresh every prompt and labels are noiseless, so learning all directions
  at once can only help.
- The literature locates Muon's cost elsewhere. Dragutinović, Zhang & Ranganath (2026) show that
  Muon gives up gradient descent's step-by-step simplicity bias. In their shared-structure
  routing task, simplified Muon memorises instead of learning the shared rule, and on a
  spurious-feature task it peaks at lower clean accuracy. Vasudeva et al. (2025) report benefits
  of spectral updates on imbalanced data. Question 4 tests the downside.

### 3. Why did rotating the input covariance do nothing for AdamW?

Xie et al. rotate the *parameter space*. Adam's per-coordinate normalisation then acts in a basis
unrelated to the network's own coordinates, in every weight matrix. We reproduced this: AdamW in a
rotated parameter space never left the plateau in 6k steps, vs step 1,900 normally. The
proposal's R rotates the *input data*, which is a different operation:

- In a GPT-2, inputs pass through the read-in W_in before any attention or MLP weight sees them.
  Rotating x → Rx is the same as replacing W_in by W_in·diag(R, 1).
- With a Gaussian init, W_in·diag(R, 1) has exactly the same distribution as W_in. The tasks
  w ~ N(0, I) are rotation-invariant as well.
- So every layer after the read-in sees a statistically identical training problem in the aligned
  and rotated conditions. The hidden coordinates are random mixtures of the input axes either way:
  from the hidden layers' point of view the "aligned" condition is rotated too.
- Only Adam's update of the read-in itself (1,152 of ~800k parameters) can tell the conditions
  apart, so the manipulation is a near-null by construction. Garg's default uniform init is not
  exactly rotation-invariant, but close.
- In the linear-attention testbed the attention weights act on the raw input coordinates, so the
  same rotation slows Adam 2.2–6.5×, as Xie-style reasoning predicts.

**Test: an axis-preserving read-in.** Initialising the read-in as the identity (input coordinate
j → residual coordinate j, `--readin_init eye`) lets the hidden layers see the data's own basis.
That is the condition under which input alignment could matter for AdamW.

Result (λ ∝ 1/i², seed 0, held-out prompts; AdamW 5e-4 for 6k steps, Muon 0.007 for 3k steps, the
tuned settings):

| optimizer | read-in | covariance | steps at which axes 1–8 are learned (probe error < 0.5) | converged | final error@32 |
|---|---|---|---|---|---|
| AdamW | identity | aligned | 425, 1,125, 1,700, 2,400, 3,200, 5,200, –, – | 5,500 | 0.043 |
| AdamW | identity | rotated | 225, 675, 1,275, 1,850, 3,400, 5,400, –, – | not within 6k | 0.048 |
| AdamW | Gaussian (confirmation, 3 seeds) | aligned | e.g. 200 … 6,200; the 6th axis at 4,200–6,700 | 4,900–5,600 (+1 never) | 0.009–0.042 at 8k |
| Muon | identity | aligned | 200, 225, 300, 400, 450, 625, 725, 900 | 775 | 0.0058 |
| Muon | identity | rotated | 225, 275, 325, 400, 550, 600, 750, 900 | 825 | 0.0059 |
| Muon | Gaussian (confirmation, 3 seeds) | aligned | the 8th axis at 750–1,025 | 750–975 | 0.005–0.007 |

Even when the data's axes reach every hidden layer, aligning the covariance gives AdamW no
advantage. The early axes even come sooner when rotated, and the differences sit well inside the
seed spread. Muon, the control, is unaffected too, and its identity-read-in runs match its
Gaussian-read-in runs. This is one seed per condition, so it rules out a large effect, not a
small one.

So the null does not come only from the random read-in. The coordinate structure Adam exploits
(Xie et al.) is the alignment of the *loss in parameter space* with the parameter coordinates. In
a multi-layer transformer, the in-context circuit is built from products such as W_Qᵀ W_K whose
other side is random at initialisation, so input-axis alignment does not make the curvature
coordinate-aligned for any single weight matrix. In the one-layer testbed, by contrast, the
trained matrix acts directly on the input coordinates, and alignment does matter (2.2–6.5×).
Practical consequence: drop the aligned-vs-rotated arm from the GPT-2 study, or keep it only as a
negative control.

### 4. Is in-context learning transient, and is Muon the most transient?

Transience cannot happen in the main setup: a fresh task in every prompt leaves nothing to
memorise. It needs a memorisable alternative (Singh et al. 2023), here a finite pool of M training
tasks (Raventós et al. 2023). The model can then either keep doing in-context regression, which
generalises, or memorise the pool, which does not.

**Design** (`--n_tasks M --noise 0.5`, `icl/transience.py`, `scripts/analyze_transience.py`):
- *Setting.* Isotropic x, d = 8, prompts of 17 points (scored on predictions from 1–16
  examples), label noise σ² = 0.25, the same GPT-2. AdamW 5e-4 vs Muon 0.005 (their tuned LRs),
  batch 64, constant LR, no weight decay for either. Both optimizers see the same task pool.
- *Pool sizes* M ∈ {16, 32, 64, 256}. These are smaller than first planned (64–4,096), because
  at M = 256 neither optimizer memorises within the budget.
- *Budget.* Seed 0 at every M for 16k steps. Seeds 1–2 were added at M = 16 and 32. Muon got
  fewer steps (5k and 12k vs AdamW's 10k and 32k), since it reaches each training loss 2–3×
  sooner. AdamW seed 0 at M = 32 was extended to 32k steps so that its fade time is not cut off.
- *References.* Two Bayes-optimal predictors are computed exactly for every evaluation prompt:
  **ridge** (posterior mean under the true prior w ~ N(0, I): in-context learning proper) and
  **dMMSE** (posterior mean when w is known to be one of the M pool tasks: memorisation).
- *ICL score* on unseen tasks: s = (E_dMMSE − E_model)/(E_dMMSE − E_ridge). 1 = generalises like
  ridge, 0 = no better than memorising.
- *Memorisation score* on pool tasks: m = (E_ridge − E_model)/(E_ridge − E_dMMSE). 1 = identifies
  pool tasks like dMMSE, 0 = no better than ridge on them. The pool error is the training loss minus
  the label-noise variance, so m is also a normalised training loss.
- *Transient ridge* (Carroll et al. 2025): s rises toward 1, then falls toward 0 while m rises.
- *Hypothesis as stated in advance.* If Muon's weaker simplicity bias makes ICL more transient,
  Muon's ICL score should fade earlier. It should do so **relative to its own emergence time**,
  not just in raw steps, since Muon is faster at everything. It should also fade at larger M than
  AdamW's. Emergence is the first checkpoint with s ≥ 0.5; fading is the first checkpoint after
  the peak with s < 0.5.
- *Counterpoint.* Small-batch noise has been reported to restore part of Muon's sequential
  learning (arXiv 2610.16868). Our Muon at batch 64 still adds input directions in eigenvalue
  order (step 3).

**Results.** There were 3 seeds at M = 16 and 32, and 1 seed at M = 64 and 256. The numbers are in
`figures/transience/transience.json`.

![ICL and memorisation scores per pool size, against training step and against each other](figures/transience/transience.png)

*Top: ICL score on unseen tasks against training step (log scale). Middle: memorisation score on
the training tasks. Bottom: the same runs with the two scores plotted against each other, i.e. the
ICL score at each level of training loss. One line per seed, smoothed over 5 checkpoints.*

Both optimizers show transient in-context learning at M = 16 and 32. The ICL score rises and
peaks, then falls as the model starts identifying the training tasks from memory. This is the
"transient ridge" of Carroll et al. (2025). At M ≥ 64, ICL persists within 16k steps for both,
although Muon's score is already declining at M = 64.

| M | optimizer | ICL emerges (s ≥ 0.5) | peak s | ICL fades (s < 0.5) | memorised (m ≥ 0.5) | fade / emerge |
|---|---|---|---|---|---|---|
| 16 | AdamW | 3,250–3,500 (2 of 3; one peaks at 0.49) | 0.49–0.52 | 3,500–3,750 | 4,000–5,000 | 1.1, 1.1 |
| 16 | Muon | 700–750 | 0.53–0.57 | 1,250–1,600 | 1,550–1,700 | 1.7–2.3 |
| 32 | AdamW | 1,250–1,300 | 0.73–0.74 | 15,250–21,250 | 16,250–20,750 | 11.7–17.0 (median 16.0) |
| 32 | Muon | 500 | 0.76–0.77 | 7,750–8,500 | 7,250–8,500 | 15.5–17.0 (median 15.5) |
| 64 | AdamW | 1,350 | 0.80 | not within 16k (s = 0.74) | not within 16k (m = 0.10) | > 11.9 |
| 64 | Muon | 550 | 0.83 | not within 16k (s = 0.63) | not within 16k (m = 0.35) | > 29.1 |
| 256 | AdamW | 2,250 | 0.86 | no (s = 0.85 at 16k) | no (m = −0.23) | > 7.1 |
| 256 | Muon | 700 | 0.90 | no (s = 0.86 at 16k) | no (m = −0.02) | > 22.9 |

*Training steps, as ranges over seeds; the last column is the pre-registered criterion.*

1. **Per step, Muon is more transient.** At M = 32 its ICL fades at 7,750–8,500 steps, against
   15,250–21,250 for AdamW, and it memorises 2.4× sooner. Under the 16k-step budget first used for
   seed 0, only Muon's ICL faded at M = 32.
2. **That is Muon's faster clock, not a different trajectory.** Muon reaches every landmark sooner
   by about the same factor. It hits the ICL landmarks (s ≥ 0.25, s ≥ 0.5) 1.9–3.0× sooner, and the
   memorisation landmarks (m ≥ 0, 0.25, 0.5) 2.2–3.0× sooner (median over seeds, smoothed curves).
   On the log-step axes, Muon's curves are AdamW's shifted left.
3. **The pre-registered criterion does not support the hypothesis.**
   - Relative to its own emergence time, Muon's ICL lasts as long as AdamW's at M = 32
     (fade/emerge 15.5–17.0 vs 11.7–17.0), and longer at M = 16 (1.7–2.3 vs 1.1).
   - The M = 16 ratio is fragile, because there ICL barely crosses the 0.5 threshold (peaks
     0.49–0.57).
   - Neither optimizer fades at M = 64 within 16k steps, so "Muon fades at larger M" is not
     observed either.
4. **At the same training loss, Muon generalises at least as well.** This analysis was added after
   seeing seed 0. Because m is a normalised training loss, comparing the ICL score at equal m
   compares the optimizers when they fit the training distribution equally well, whatever their
   speed.

| M | m = −0.5 | m = 0 | m = 0.25 | m = 0.5 |
|---|---|---|---|---|
| 16 | 0.30 vs **0.40** | 0.42 vs **0.50** | 0.48 vs **0.53** | 0.46 vs 0.48 |
| 32 | 0.60 vs **0.67** | 0.71 vs **0.75** | 0.66 vs 0.67 | 0.51 vs 0.51 |
| 64 (1 seed) | 0.72 vs 0.73 | 0.77 vs 0.80 | – | – |
| 256 (1 seed) | 0.78 vs 0.79 | – | – | – |

*ICL score on unseen tasks, AdamW vs Muon (mean over seeds), at the first moment each run
reaches a given memorisation score. Bold: seed ranges do not overlap. m = 0 is where the model
first beats ridge on the training tasks. "–": not reached by both optimizers.*

   - Before memorisation takes over (m ≤ 0), Muon's ICL score is 0.04–0.10 higher at M = 16
     and 32, with separate seed ranges. At M = 64 and 256 it is 0.01–0.04 higher (one seed).
   - Put the other way: at the same training loss, AdamW generalises worse to new tasks. More of its
     progress comes from what is specific to the training tasks.
   - Once memorisation dominates (m ≥ 0.5), the two coincide.
   - Muon's peak ICL score is also higher at every M: 0.53–0.57 vs 0.49–0.52 at M = 16, and
     0.76–0.77 vs 0.73–0.74 at M = 32.
   - In the bottom row of the figure, the optimizers trace nearly the same path from ridge towards
     dMMSE; Muon travels it faster.
   - The comparison barely depends on smoothing: windows of 1, 3 or 5 checkpoints move it by
     ≤ 0.02 (checked on seed 0).

**Verdict: the hypothesis is not supported here.**
- Muon's in-context learning fades sooner per step, because Muon learns everything about 2.5×
  faster.
- It is not more transient. Relative to its own clock it lasts as long as AdamW's. At the same
  training loss it generalises at least as well, and better early on.
- The prediction that weaker simplicity bias makes Muon memorise sooner does not show up.

Possible reasons, none tested here:
- Practical Muon keeps much of the learning order. In this GPT-2 it still adds input directions in
  eigenvalue order, only faster (step 3). Dragutinović et al. studied a simplified Muon.
- Minibatch noise (batch 64) may restore part of the sequential behaviour (arXiv 2610.16868).
- Both optimizers follow the same path, so the switch from ridge to dMMSE looks like a property of
  the training distribution, with the optimizer setting only the speed along it. That fits the
  loss–complexity account of transience (Carroll et al. 2025; Wurgaft et al. 2025): the memorising
  solution takes over once its lower loss outweighs its complexity.

**Consequences for the study.**
- Compare transience at matched training loss, or on a log-step axis, not at a fixed step count.
  With a fixed budget, Muon looks more transient than it is.
- For persistent ICL with Muon, use larger pools (M ≥ 256 here) or stop earlier. At M = 32 Muon's
  ICL window (s ≥ 0.5) lasts about 7k steps, against 14k–20k for AdamW.
- SGD cannot take part in this test in this GPT-2 (question 1).

**Limitations.**
- One LR per optimizer (the tuned values from the noiseless study), a constant LR, and d = 8.
- No weight decay. Singh et al. (2023) found that L2 regularisation can make ICL persist, so weight
  decay could interact with the optimizer.
- Three seeds at M = 16 and 32, but only one at M = 64 and 256.
- "Muon" includes AdamW at lr 1e-3 on the embeddings, read-in/out and norms, as in the main study.
