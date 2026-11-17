## Verification step 1 — can this setup reproduce the papers the hypotheses rest on?

Each paper's central claim was re-derived in our code, either exactly (theory papers: no sampling
noise, gradient flow integrated with an adaptive ODE solver) or in our GPT-2 (Garg, Xie). These
are positive controls: if an instrument cannot reproduce a known result, its new results
should not be trusted.

| Paper | Claim tested | Our result | Reproduced? |
|---|---|---|---|
| **Zhang et al. 2025** | Separate rank-1 K/Q linear attention under gradient flow learns principal axes one at a time, in eigenvalue order. Plateau losses are L_m = trΛ − Σ_{i≤m} λ_i / (1 + (1 + trΛ/λ_i)/N). Merged K·Q makes one drop to the optimum. | All 16 predicted plateaus (λ ∝ 1/i and 1/i²) match to 5–6 decimals, e.g. 0.393225 measured vs 0.393226 predicted. Axes switch on in eigenvalue order, with turn-on time ∝ λ^−2.06 under 1/i². Both parametrizations reach the analytic optimum (relative error ~1e-14). | **Yes, exactly** |
| **von Oswald et al. 2023** | A trained linear self-attention layer implements one (preconditioned) gradient-descent step. | The merged model's converged Γ equals the analytic one-step preconditioned GD Γ* = U diag(N/((N+1)λ_i + trΛ)) Uᵀ to 7e-16 (isotropic) and 4e-14 (1/i²). | **Yes, exactly** |
| **Ma et al. 2026** | Simplified Muon minimises f(Q) = ½ tr((SQ − I) S (SQ − I)ᵀ) at a κ-independent rate; GD's rate depends on κ; SignGD equals Muon only when S is axis-aligned. | Iterations to 1e-6 as κ(S) goes 1 → 16: GD 1 → 17,852 (∝ κ³); Muon 45–68 for every κ ≥ 2, aligned or rotated. Aligned SignGD equals Muon iteration-for-iteration; rotated SignGD needs 343 iterations at κ = 4, 2,657 at κ = 8, and fails at κ = 16. | **Yes** |
| **Saxe et al. / Dragutinović et al. 2026** | In a deep linear network GD learns singular modes sequentially (time ∝ 1/s); spectral GD learns them together. | GD 30 → 640 steps from the largest to the smallest mode, identical when rotated. Spectral GD 70 → 20 steps, so small modes finish *first*, and identical when rotated. Adam is parallel when the modes are axis-aligned and sequential (GD-like) when they are rotated. | **Yes** |
| **Garg et al. 2022** | A GPT-2 trained on isotropic inputs matches least squares in context and beats 3-NN and averaging. Under a skewed covariance, its error stops improving while least squares goes to 0. It tracks least squares under label noise (no peak at k = d), predicts ≈0 information for queries orthogonal to the context, and copies queries that repeat a context point. | Our isotropic-trained GPT-2s match all of these (figure below): least squares in distribution (error ≤0.017 for k ≥ 16; 3-NN 0.59); ≈ the Bayes ridge estimator under noise, avoiding least squares' blow-up at k = 8; error ≈ 1.0 on orthogonal queries; ≈ 0.01 on copied queries. The skewed-covariance failure reproduces for the **AdamW-trained** model, whose error *rises* from 0.21 (k = 7) to 0.47 (k = 32). | **Yes** |
| **Xie et al. 2025** | Rotating the full parameter space slows Adam sharply (SGD is invariant by construction). | In our GPT-2 (isotropic, lr 3e-4, same seed and data), AdamW in a randomly rotated parameter space never leaves the plateau in 6k steps, vs step 1,900 unrotated. | **Yes, strongly** |
| **Garg et al. 2022 — training recipe** | Their curriculum (dimensions 2 → 8 here, with 4d+1 points) speeds up training, and their own Adam-trained model (lr 1e-4, batch 64) shows the skewed-covariance failure. | Same recipe, same seed, Garg's token layout, with vs without curriculum: error 0.204 vs 0.356 at step 8k; both ≈0.155 by 20k. The curriculum model's skewed error stops improving at k ≈ 7 and drifts up (0.21 → 0.32 → 0.24) while least squares reaches 0. The *no-curriculum* model degrades less (0.24 → 0.09, still falling). So the size of Garg's failure depends on training details. | **Yes**, with a nuance |

![Zhang et al. reproduction](figures/verify/repro_zhang.png)
![Ma et al. reproduction](figures/verify/repro_ma.png)
![Garg et al. evaluation suite on our isotropic-trained models](figures/verify/garg_eval_pilot_iso.png)
![Garg et al. curriculum](figures/verify/repro_garg_curriculum.png)
![Deep linear network: GD vs spectral GD vs Adam](figures/verify/repro_deeplin.png)
