## What changed after verification

The pilot's GPT-2 numbers came from learning rates tuned only on isotropic inputs, mostly one seed,
and evaluation on the prompts used for tuning. The verification pass (steps 1–3 below) redid this
properly. These pilot statements changed:

1. **Both adaptive optimizers were under-tuned on anisotropic inputs.** AdamW's best LR is 5e-4 on
   every condition (the pilot used 3e-4). Muon's is 0.005 on isotropic inputs (unchanged) and
   0.007 on λ ∝ 1/i² (the pilot used 0.005). β2 = 0.95 vs 0.999 made no difference.
2. **Muon's isotropic speed-up is real but smaller:** 2.7× to leave the plateau and 4.3× to
   converge, not ~5×. Under 1/i² it is 6–7×, close to the pilot's 5–7×.
3. **AdamW on λ ∝ 1/i² is seed-sensitive.** One of three aligned seeds did not converge within
   8k steps. Two of six 1/i² runs end the LR decay (10k steps) still ignoring the weakest axis
   (implied-weight gain ≈ 0), i.e. stuck at a truncated, PCR-like solution. So "all optimizers
   end near least squares" (H4) holds in distribution but not on every axis within this budget.
4. **H3 is firmer than the pilot said, but its end-state size is seed-driven.** At matched
   in-distribution error Muon's shifted-query error is 1.5–2.5× lower, aligned and rotated (the
   pilot had ≈2× aligned and ≈1.15× rotated), and the gap does not close within the budget. The
   large end-of-training gaps (0.04 vs 0.34) come mostly from the AdamW seeds that never learned
   the weakest axis.
5. **"Muon learns all directions in parallel" is too strong.** Muon also adds axes in eigenvalue
   order, but compresses the staircase into ~1,000 steps, and a ridge filter fits it better almost
   throughout. AdamW's PCR staircase survives tuning and fresh seeds (PCR-like for 79–96% of
   learning, every seed).
6. **SGD's failure now has a measured cause.** On the plateau the query/key gradient is ~10⁻⁴ of
   the total, so the attention circuit never starts. It fails even at d = 2, with Garg's
   curriculum, and with clipping up to lr 1.0. This is an optimizer × architecture effect, so H2
   and H4 can only be tested with AdamW in SGD's place in this model.
7. **The pilot's layer-specific head-alignment observations do not replicate** across fresh seeds.
   One pattern does: Muon's layer-2 heads cover every axis, and AdamW's give the weakest axis zero
   share (6/6 runs).
8. **Garg's skewed-covariance failure reproduces, but its size depends on training details**
   (curriculum vs none). Muon-trained models were robust to it in 5/5 cases vs 2/6 for Adam-trained
   ones. That is a new, suggestive observation, not an established effect.

These held: Muon's faster emergence (H1), the imbalance penalty for AdamW but not Muon (H2), the
AdamW staircase vs Muon ridge-like contrast (H4 as revised), the null effect of rotating the input
covariance for AdamW, and every theory-testbed conclusion (now also with exact gradients, dense LR
grids and two init scales). bf16 numerics are not a confound.
