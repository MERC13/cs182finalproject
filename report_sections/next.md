## Recommended changes for the full study

*Update:* the section "A better experiment, given the negative results" supersedes items 1, 2 and 12 where they conflict.
- SGD can be trained in the GPT-2 with concatenated tokens and hidden init 1/√width.
- The rotation arm should be a negative control.
- Transience is demoted.
- The recommended study is optimizer × batch size × input spectrum, read out as the spectral learning exponent.

1. **Decide what replaces SGD.** SGD cannot form the ICL circuit in this GPT-2 (step 2), so H2 and
   H4 as written cannot be tested on it. Options, best first:
   - (a) State H2 and H4 for "non-orthogonalised optimizers", with AdamW as the comparison.
   - (b) Test SGD where it does learn: the linear-attention testbed, where its staircase is
     predicted and observed.
   - (c) ~~SGD with per-block learning rates~~ was tested in the follow-up (question 1) and does
     not work. Normalised SGD, with Muon's exact per-matrix step size, still never leaves the
     plateau. So does SGD on the hidden matrices with AdamW on everything else. If an
     "SGD-family" arm is wanted, normalised SGD is the informative one: it is Muon without the
     orthogonalisation.
2. **Use the corrected token layout** (y in its own coordinate, Gaussian read-in init; `--y_mode sep
   --readin_init gauss`). Add **Muon with the read-in included**, the only exactly
   rotation-invariant Muon arm, if the aligned-vs-rotated contrast stays in the study. Otherwise
   drop the rotation: it is a confirmed null for AdamW here, and only a full parameter rotation
   (Xie et al.) moves it. The follow-up (question 3) found no effect even with an identity
   read-in that hands the input axes to every hidden layer, so dropping it is the better choice.
3. **Tune per condition and select on separate prompts.** Start from the verified values (AdamW
   5e-4; Muon 0.005 isotropic, 0.007 under 1/i²; aux AdamW 1e-3). Sweep at least a decade below
   the usual defaults: AdamW stalls at 3e-3, Muon at 0.02. Rank by area under the learning curve
   and keep a held-out prompt set for the final numbers.
4. **Use a constant-then-decay schedule with branches.** Read timing (H1, H2) from the
   constant-LR phase and end states (final error, H3, Garg's suite) from decay branches. Resuming
   is bit-exact in this code (`--resume`, `--schedule decay`).
5. **Train much longer for end-state claims.** Two of six AdamW 1/i² runs still ignored the
   weakest axis after 10k steps. Whether AdamW eventually reaches least squares on every axis
   decides whether H3 is a lasting difference or a transient one, as in the testbed. Budget
   50k–100k+ steps on a GPU.
6. **Seeds:** ≥ 3 per cell, and ≥ 5 for AdamW on skewed spectra, whose weak-axis times and
   end states vary several-fold across seeds.
7. **Measure per-direction learning with implied-weight gains**, not only probe tasks; probe tasks
   are out of distribution in scale for weak axes. Test H3 at **matched in-distribution error**,
   and report how many seeds reach each level.
8. **Run Garg et al.'s full evaluation suite on every final model.** It is cheap. With more seeds,
   it can test the new observation that Muon-trained models resist the skewed-covariance failure.
9. **Keep the reproductions as regression tests** for the full-scale code: Zhang's plateau values,
   the exact equivariance checks, Xie's parameter-rotation control, and Garg's suite. If a code
   change breaks one, the new results should not be trusted.
10. **Reference estimators:** least squares, the fitted ridge path, Bayes-PCR(m), and Zhang et
    al.'s finite-N one-step preconditioned GD. Consider one run family with label noise σ > 0,
    where "ridge-like" differs from least squares in a principled way (Raventós et al.'s
    setting).
11. **Tighten the background section** (see the literature table):
    - Zhang's staircase needs rank-1 heads and small init.
    - Xie did not run SGD rotated.
    - Aranda et al. is a workshop paper comparing Muon with Adam and NMD.
    - Ma et al.'s result is for a one-layer linear model with exact orthogonalisation and a
      decaying step size.
12. **If transience is part of the study** (follow-up question 4):
    - Use a finite task pool with label noise, scored against exact ridge and dMMSE references
      (`--n_tasks`, `--noise`; `icl/transience.py`).
    - Compare optimizers at matched training loss or on a log-step axis, never at a fixed step
      count. Muon runs the same trajectory about 2.5× faster, so a fixed budget makes it look more
      transient.
    - At this scale, pools of 16–32 tasks show the full transition within 32k AdamW steps. At 64
      tasks only Muon has started to decline by 16k steps, and at 256 neither has.
    - Add weight decay as a factor (Singh et al.), and use ≥ 3 seeds.
