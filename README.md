# icl_optim — pilot code for "Does the optimizer change how ICL is learned?"

Initial tests of the CS 182 proposal (Chu, Dao, Hu, Cha). **Results: [`REPORT.md`](REPORT.md).**

## Final report: figures, statistics and checkpoints

Everything in the final report's results comes from the committed JSON logs; no retraining is needed.

```bash
python3 scripts/report_data.py          # Figures 1-3 data -> figures/report/*.json
python3 scripts/analyze_garg_seeds.py   # Section 4 statistics (three seeds, matched errors)
python3 scripts/analyze_garg_scale.py   # seed-10 detail, including exact Muon
python3 scripts/ood_map_linattn.py      # linear attention: query shifts at matched error
python3 scripts/analyze_schatten_pilot.py  # linear attention: the spectral knob
python3 scripts/analyze_garg_lr.py      # learning-rate check at Garg scale (Appendix B)
python3 scripts/analyze_algorithm.py    # H1 algorithm: distance to nearest PCR vs nearest ridge (Table 2)
```

**Checkpoints.** `checkpoints/<dir>/<run>.pt` holds the final float32 weights of every GPT-2 run the report uses
(`garg_scale/`: 12 layers, width 256, seeds 10-12, plus exact Muon; `init/`, `knob/`, `ood/`: the smaller GPT-2),
with the run's training flags. Load one and check it against its log:

```bash
python3 scripts/eval_checkpoint.py checkpoints/garg_scale/muon_s11.pt   # or --all
```

```python
from scripts.eval_checkpoint import load_model
model, config, ck = load_model("checkpoints/garg_scale/muon_s11.pt")   # model(x, y) -> predictions
```

The weights are those at each run's last evaluation; the matched-error measurements in the report are taken at
earlier steps and are read from the logs. Retrain any run with `python3 scripts/run_queue.py jobs/<file>.txt`.
The linear-attention runs need no checkpoints: their learned matrix at every logged step is in
`runs/linattn/ood_gamma.json`.

A small re-implementation of the proposal's setup in plain PyTorch:
- Garg et al. (2022)-style in-context linear regression;
- a GPT-2 identical in architecture to Garg's `TransformerModel`;
- SGD / AdamW / Muon optimizers;
- a one-layer linear-attention "theory testbed".

It runs on CPU (as in the pilot) or on a GPU (`--device cuda`).

```
icl/data.py        covariance conditions (lam_i ∝ i^-alpha, trace d; aligned or Haar-rotated),
                   on-the-fly sampler (optionally a finite task pool + label noise); fixed test /
                   probe / query-shift / implied-weight prompt sets
icl/model.py       GPT-2 (pre-LN, learned positions, gelu_new MLP, dropout 0), linear read-in and
                   read-out, separate Q/K/V; --y_mode garg|sep|concat, --readin_init torch|gauss|eye,
                   --hidden_init (one std for all hidden matrices), --freeze_readin
icl/optim.py       SGD+momentum, AdamW, Muon (reference NS5 / exact polar, Nesterov, shape scaling)
                   with auxiliary AdamW for non-hidden parameters; --muon_orth fro|none ablations
                   (normalised SGD / plain SGD on the hidden matrices)
icl/estimators.py  OLS (min-norm), ridge grid, Bayes-PCR(m), one-step preconditioned GD
                   (Zhang et al. finite-N form), one-step GD
icl/train.py       training loop with periodic evaluation; one JSON log per run
icl/transience.py  finite-pool references (ridge, dMMSE) and the transience evaluation (+ Zipfian pools, --zipf)
icl/lintf.py       multi-layer linear transformers on concatenated tokens (the model ladder)
icl/linattn.py     linear-attention testbed (separate rank-1 K/Q or merged KQ), exact population loss
icl/heads.py       head alignment through the input embedding (exploratory)
icl/repro_theory.py exact reproductions of Zhang et al., von Oswald et al., Ma et al., deep-linear theory
icl/analysis.py    log loading and metrics
scripts/           run_queue.py (parallel, restart-safe, checkpoint-aware job runner), summarize.py,
                   progress.py, search_report.py, confirm_jobs.py, analyze_confirm.py, garg_eval.py,
                   analyze_grid.py, gpt2_table.py, heads_report.py, analyze_linattn.py,
                   analyze_exact_sweep.py, plot_pilot.py, plot_sweep.py, plot_repro.py, build_report.sh;
                   follow-up: grad_blocks.py, grad_spectrum.py, analyze_eye.py, analyze_transience.py;
                   redesign pilots: noise_sweep_linattn.py, ladder_sweep.py, analyze_noise.py,
                   analyze_ladder.py, analyze_zipf.py
jobs/              the exact job lists the pilot ran (regenerated from the run logs)
runs/              JSON logs (+ final .pt models for the grid);  figures/  all plots
```

## Setup

```bash
pip install torch numpy scipy matplotlib
```

## Reproduce the verification study

```bash
python -m icl.repro_theory --out figures/verify/repro_theory.json          # Zhang, von Oswald, Ma, deep linear (exact)
python -c "from icl.linattn import exact_sweep; exact_sweep('runs/linattn/exact_sweep.json')"
python scripts/analyze_exact_sweep.py runs/linattn/exact_sweep.json
python scripts/run_queue.py jobs/search_b1.txt --procs 4                   # per-condition hyperparameter searches
python scripts/run_queue.py jobs/search_b2.txt --procs 4                   # (+ Garg-recipe and Xie reproductions)
python scripts/search_report.py                                            # rank configurations by AUC
python scripts/confirm_jobs.py --adamw iso=5e-4:0.999,ani=5e-4:0.999 --muon iso=0.005,ani=0.007 > jobs/c.txt
python scripts/confirm_jobs.py --adamw iso=5e-4:0.999,ani=5e-4:0.999 --muon iso=0.005,ani=0.007 --stage decay > jobs/d.txt
python scripts/run_queue.py jobs/c.txt --procs 4 && python scripts/run_queue.py jobs/d.txt --procs 4
bash scripts/make_confirm_figures.sh        # confirmation table, figures, Garg suite, skewed summary, head alignment
python scripts/plot_repro.py figures/verify  # reproduction and confound-check figures
bash scripts/build_report.sh                 # report_sections/*.md -> REPORT.md
```

## Reproduce the follow-up (why SGD fails, identity read-in, transience)

```bash
python scripts/grad_blocks.py                # per-block plateau gradients and per-step changes
python scripts/grad_spectrum.py              # singular spectra of the hidden-matrix gradients
python scripts/run_queue.py jobs/followup.txt --procs 4         # normalised SGD, SGD-hidden, identity read-in, transience seed 0
python scripts/run_queue.py jobs/transience_seeds.txt --procs 4 # transience: seeds 1-2, AdamW M=32 to 32k
python scripts/analyze_eye.py
python scripts/analyze_transience.py figures/transience 'runs/transience/*.json' 'runs/transience_ext/*.json'
```

Checkpoints: every run saves a resumable `.ckpt` (model, optimizer state, data-stream RNG, log);
`--resume X.ckpt --steps N --schedule decay --decay_steps D` branches a learning-rate decay from it.
`run_queue.py` starts a `--resume` job only once its checkpoint exists.

## Reproduce the pilot

```bash
python -m icl.linattn --out runs/linattn/grid.json --steps 4000 --seeds 3        # testbed, ~1 CPU-hour
python scripts/run_queue.py jobs/pilot_sweep.txt --procs 4                        # GPT-2 LR sweep (iso)
python scripts/run_queue.py jobs/pilot_grid.txt --procs 4                         # GPT-2 main grid
python scripts/analyze_linattn.py runs/linattn/grid.json figures/linattn
python scripts/analyze_grid.py 'runs/grid/*.json' figures/grid
python scripts/plot_pilot.py 'runs/grid/*.json' figures/grid
python scripts/plot_sweep.py 'runs/sweep/*.json,runs/plateau/*sep*.json' figures/grid/lr_sweep_iso.png
python scripts/heads_report.py 'runs/grid/*.json' figures/grid
scripts/build_report.sh                                                           # report_sections/ -> REPORT.md
```

One run:

```bash
python -m icl.train --opt muon --lr 0.005 --cond 1/i2-ali --y_mode sep --readin_init gauss \
    --steps 5000 --device cuda --out runs/demo.json
python scripts/summarize.py runs/demo.json
```

Conditions: `iso`, `1/i-ali`, `1/i-rot`, `1/i2-ali`, `1/i2-rot`. Pilot LRs, tuned on `iso`:
- Muon 0.005 (aux AdamW 1e-3);
- AdamW 3e-4;
- SGD: no stable LR left the plateau; 0.1 with `--clip 1.0` was used.

## What is logged (every `--eval_every` steps, on fixed test prompts)

* `test_err_k`: error at every prefix length k = 0..32, normalised so that predicting 0 scores 1.
* `probe_err_k`: relative error on prompts whose task lies along one principal axis.
* `dist_1_32`, `dist_8_32`: squared distance between the model's predictions and each reference
  estimator's (OLS, ridge λ grid, PCR m = 1..8, gdpre m = 1..8, gd1).
* every `--qshift_every` steps:
  - `qshift_err`: the query-shift test, t = 0..1. Tasks and contexts are kept, and query variance
    moves onto the low-variance axes at fixed total variance.
  - implied-weight fit: `iw_gain` and `iw_err` per principal axis, plus `iw_r2` as a linearity
    check.

## Design choices worth knowing

* **Token layout.** `--y_mode sep --readin_init gauss` (the pilot default) gives y its own input
  coordinate and a rotation-invariant read-in init. With that, SGD — and Muon if it also updates
  the read-in (`--muon_params hidden+in`) — is exactly equivariant to rotating the inputs. Only
  AdamW, and Muon's AdamW-updated read-in, can tell aligned from rotated. Garg's layout
  (`--y_mode garg`) breaks this symmetry.
* **Precision.** bf16 autocast for linear layers is used on CPU only (`--bf16 1`; attention stays
  fp32). GPUs run in fp32. Evaluation is always fp32. On one Sapphire Rapids core a step at batch
  64 takes ~0.22 s.
* **Weight decay** is 0 for every optimizer by default.

## Reproduce the redesign pilots (report section "A better experiment, given the negative results")

```bash
python scripts/noise_sweep_linattn.py runs/linattn/noise_sweep.json --procs 4    # testbed: batch 16-1024 x GD/Adam/SignGD/Muon
python scripts/noise_sweep_linattn.py runs/linattn/noise_sweep_muonvar_m0.json --opts muon,muon_exact --conds 1/i2-ali --muon_momentum 0
python scripts/noise_sweep_linattn.py runs/linattn/noise_sweep_muonvar_exact.json --opts muon_exact --conds 1/i2-ali
python scripts/run_queue.py jobs/noise_gpt2_a.txt --procs 4    # GPT-2 at batch 256 (AdamW aligned / rotated); stopped early in the pilot
python scripts/run_queue.py jobs/noise_gpt2_b.txt --procs 4    # GPT-2 Muon at batch 16 / 256; Zipfian pool
python scripts/run_queue.py jobs/concat_then_rest.txt --procs 4   # concatenated tokens (SGD, AdamW, Muon)
python scripts/run_queue.py jobs/sgd_init.txt --procs 4        # concatenated tokens + hidden init 1/sqrt(128)
python scripts/run_queue.py jobs/rescue.txt --procs 4          # the redesigned GPT-2 under λ ∝ 1/i² (seed 10); jobs/rescue_s11.txt: seed 11
python scripts/ladder_sweep.py runs/ladder/ladder.json --procs 4
python scripts/analyze_noise.py figures/noise
python scripts/analyze_ladder.py runs/ladder/ladder.json figures/ladder
python scripts/analyze_zipf.py 'runs/zipf/*.json'
```
