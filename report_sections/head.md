# Does the optimizer change how in-context learning emerges? Pilot and verification

*Initial tests of the CS 182 proposal by Chu, Dao, Hu & Cha, followed by a verification pass:
reproductions of the cited papers, per-condition hyperparameter search, and confirmation runs on
fresh seeds with held-out prompts. CPU-only, November 2026. Code in `icl_optim/`, figures in
`icl_optim/figures/`.*

**Bottom line (after verification).** The proposal's central claim survives, with smaller and
more qualified effects than the pilot reported.
- In the 4L/4H/128d GPT-2, Muon learns in-context linear regression faster than AdamW: 4.3×
  faster to converge on isotropic inputs and 6–7× under λ ∝ 1/i².
- Under 1/i², Muon learns the weakest input direction 7× sooner. At matched in-distribution
  error it is 1.5–2.5× more accurate on queries that load the weak directions.
- The mechanism claims needed revising. The PCR staircase belongs to AdamW, not SGD. Muon runs a
  compressed, ridge-like progression, not a truly parallel one.
- SGD cannot be tested in this model at all. It never leaves the plateau at any setting tried.
  On the plateau its query/key gradients are ~10⁻⁴ of the total, so the attention weights never
  move.
- Rotating the input covariance changes nothing for AdamW. That holds even with an identity
  read-in that hands the input axes to every layer (follow-up). The instrument can see basis
  effects: rotating AdamW's whole parameter space (Xie et al.) stops learning entirely.
- Before any of this, the code reproduced seven published results, several to 5–6 decimals.

**Follow-up questions** (section "Follow-up: four questions after verification"):
- *Why SGD fails.* There are two measured causes.
  - One global LR cannot serve blocks whose per-weight gradients differ ~7,000×.
  - Each hidden matrix's plateau gradient is nearly rank-1.
  - Normalised SGD with Muon's exact per-matrix step size still never learns, so
    orthogonalisation is what lets Muon escape.
- *The takeaway is narrower than "Muon is better for skewed covariance".* Muon learns the weak
  directions nearly together where AdamW learns them one at a time. That is evidence from one
  small model with nothing to overfit, not a proof.
- *Rotation.* Rotating the inputs is not the parameter-space rotation that hurts Adam. Drop the
  aligned-vs-rotated arm or keep it as a negative control.
- *Transience.* With a finite pool of training tasks, both optimizers show transient ICL
  (M ≤ 32 tasks).
  - Muon's ICL fades ~2.5× sooner in steps, but only because Muon does everything ~2.5× faster.
  - Relative to its own clock it lasts as long as AdamW's. At the same training loss it
    generalises as well or better.
  - "Muon is the most transient" is not supported.

**A better experiment** (section "A better experiment, given the negative results"; literature review plus pilots):
- *SGD can be trained in the GPT-2.* It needs two changes together: concatenated (x, y) context tokens and a larger init for the hidden matrices (std 1/√width). Either change alone leaves SGD stuck.
- *In that model the proposal's ordering appears.* Under λ ∝ 1/i², SGD, AdamW and Muon learn the input directions with spectral learning exponents 1.11 > 0.80–0.91 > 0.51–0.60 (time to learn direction j ∝ λ_j^−β; two seeds, no overlap). SGD is the most sequential and Muon the least.
- *Gradient noise is the lever on Muon's simplicity bias.* Its exponent rises from 0.25 to 0.45 in the GPT-2 (batch 256 → 16, one seed each) and from 0.37 to 0.90 in the one-layer testbed (batch 1,024 → 16), as Jacobs & Burkholz (2026) predict.
- *Rotation stays null at batch 256 too.* Rotating the inputs is an output-wise rotation of the read-in, the kind Adam tolerates (Zhang, Maes et al. 2024).
- *Recommended study:* optimizer × batch size × input spectrum in the redesigned GPT-2, read out as the spectral learning exponent and matched-error OOD error. Keep rotation only as a negative control.

| Hypothesis | Pilot claim | Verified result (tuned per condition, 3 fresh seeds, held-out prompts) | Status |
|---|---|---|---|
| **H1** Muon reaches low unseen-function error earlier | Muon converges 5–7× sooner than AdamW under every covariance; SGD never learns. | Isotropic: escape 433 vs 1,167 steps (2.7×), converged 1,392 vs 5,933 (4.3×). λ ∝ 1/i²: converged 867 vs 5,250 aligned (6×; one AdamW seed not within 8k) and 888 vs 6,200 rotated (7×). Seed ranges never overlap. | **Confirmed**; smaller on isotropic inputs (4.3×, not ~5×) |
| **H2** imbalance slows SGD more than Muon | SGD unscorable. With AdamW in its place, the weakest axis takes 2.5k → 7.8–8.5k steps (iso → 1/i²) vs 0.5k → 0.9k for Muon. | AdamW 1,550 → 6,350 steps (one seed never in 8k); last-axis / first-axis time ratio 1.0 → 37. Muon 550 → 900; ratio 1.0 → 4.3. | **Confirmed for AdamW. Untestable for SGD** in this model |
| SGD arm | Never leaves the plateau (10k steps). | Fails at every LR / momentum / clipping setting tried, at d = 2 (AdamW learns in 500 steps), and with Garg's curriculum. Measured cause: query/key gradient 9×10⁻⁵ vs total 0.29; the attention weights do not move. | **Confirmed, with a cause**: optimizer × architecture, not input variance |
| **H3** smaller error increase when queries load low-variance directions | ≈2× (aligned) and ≈1.15× (rotated) at matched in-distribution error; likely transient. | At matched in-distribution error 0.03 / 0.01 / 0.005, Muon's shifted-query error is 1.5–1.6× / 1.6–2.2× / 2.5× lower, aligned and rotated. The gap does not close within the budget. | **Confirmed**, stronger than the pilot's rotated estimate. The end-state gap is driven by AdamW seeds that never learn the weakest axis |
| **H4** SGD → PCR on a growing set; Muon → ridge-like; all end near least squares | The staircase belongs to AdamW (rank 1 → 8, ~1 rank per 1.2k steps). Muon ridge-like, all axes within 3×. | AdamW under 1/i² is PCR-like for 79–96% of learning, every seed (path ratio 0.76 aligned, 0.80 rotated). Muon is ridge-like in every seed (ratio 1.11–1.25). It adds axes in eigenvalue order too, but within ~1,000 steps. 2 of 6 AdamW runs end (10k steps) with gain ≈ 0 on the weakest axis. | **Revised**: staircase = AdamW; Muon = compressed and ridge-like, not parallel; "all end near least squares" holds in distribution only |
| Design: aligned vs rotated covariance | Large in the testbed (Adam 2–6.5× slower rotated); no effect for AdamW in the GPT-2. | AdamW converged at 5,250 vs 6,200 steps, error@32 0.021 vs 0.023, PCR ratio 0.76 vs 0.80: all within seed noise. A full parameter-space rotation (positive control) stops AdamW from learning at all. | **Confirmed null** for the GPT-2, also with an identity read-in (follow-up, one seed) |
| Added: Garg's skewed-covariance test (not in the proposal) | The AdamW-trained model's error rises with k; Muon's does not (one model each). | Robust (error keeps falling with k): Muon 3/3 confirmation models, 5/5 overall; AdamW 1/3, 2/6 Adam-trained models overall. | **Suggestive only**: small n, and the Adam models were trained under mixed settings |

**How the findings were verified** (sections below):
1. *Reproductions.* Zhang et al.'s plateau law, von Oswald et al.'s one-step GD solution, Ma et
   al.'s κ-independent Muon rate, sequential vs spectral learning in deep linear networks, Garg et
   al.'s evaluation suite and curriculum, and Xie et al.'s rotated-Adam slowdown, all in our code.
2. *Confounds.* Per-condition LR search on separate selection prompts; a fair chance for SGD (LR,
   momentum, clipping, d = 2, curriculum); fp32 vs bf16; a positive control showing the GPT-2 is
   sensitive to Adam's coordinate system.
3. *Confirmation.* A protocol fixed in advance: tuned configs, fresh seeds 10–12, a held-out
   prompt set, and an LR-decay branch so end-state claims are not read off a constant-LR noise
   floor.

**Limitations.**
- CPU only, so runs are 4k (Muon) to 10k (AdamW) steps on a 0.8M-parameter model. Garg et al.
  trained 500k steps. Whether AdamW eventually learns its weakest axes is untested.
- 3 seeds per cell (2 for Muon rotated). The confirmation covers isotropic and 1/i² inputs only;
  the 1/i condition is pilot-only.
- "Muon" means Muon on the hidden matrices plus AdamW on embeddings, read-in/out, biases and norms,
  as in the reference implementation.
- AdamW got 2.5× more steps than Muon, which favours AdamW in every end-state comparison.

**Files.**
- `REPORT.md` — this report.
- `figures/verify/` (reproductions, confound checks) and `figures/confirm/` (confirmation runs);
  pilot figures in `figures/grid/` and `figures/linattn/`.
- `runs/` — JSON logs of every run (`runs/confirm/` holds the confirmation logs and checkpoints).
- `icl/` and `scripts/` — the code (see `README.md`).
