# How Optimizers Shape In-Context Learning

Code and data for our CS 182 project paper, *How Optimizers Shape In-Context Learning*. We train a GPT-2 and one-layer
linear attention on in-context linear regression with SGD, AdamW and Muon, and compare the in-context algorithm each
one learns.

```bash
git clone --depth 1 https://github.com/MERC13/cs182finalproject   # the full history adds about 1 GB of pilot logs
pip install -r requirements.txt
```

## Reproduce the paper

Every figure and number is computed from the committed logs in `runs/`; this takes about a minute.

```bash
python scripts/figures.py        # figures/*.pdf
python scripts/results.py        # every number in the text and tables
```

## Retrain

```bash
grep -v '^#' jobs.txt | xargs -P 4 -L 1 python -m icl.train   # 35 GPT-2 runs, 1-5 CPU hours each
python -m icl.linattn runs/linattn.json                       # linear attention, about 5 minutes on 4 cores
python -m icl.linattn runs/linattn_rotated.json --rotated     # the same with the inputs in a random basis
```

Training is deterministic on one CPU thread: with the pinned versions, reruns reproduce the logged values exactly on
our machine (other CPUs can differ in the last digits). The exception is two runs that originally used several threads
(`garg_scale/muon_pow1_lr0.0005_s10`, `garg_lr/muon_pow1_lr0.007_s10`), whose reruns match to rounding at first and
then drift. The final weights of every GPT-2 run are next to its log:

```python
from icl.model import GPT2
model = GPT2.load("runs/garg_scale/muon_s10.pt")   # model(x, y) -> predictions
```

## Layout

| Path | Contents |
| --- | --- |
| `icl/data.py` | the task and the evaluation prompts (in-distribution, probe, shifted queries, gain fits) |
| `icl/model.py` | the GPT-2 |
| `icl/optim.py` | Muon and the spectral knob |
| `icl/estimators.py` | reference estimators: ridge, PCR, one-step gradient descent |
| `icl/train.py` | GPT-2 training and evaluation |
| `icl/linattn.py` | linear attention: model, exact losses, learning-rate sweep |
| `icl/analysis.py` | reading the logs at a matched error |
| `scripts/` | `figures.py` and `results.py` |
| `figures/` | the paper's figures |
| `runs/garg_scale/` | logs (`.json`) and final weights (`.pt`) of the 12-layer GPT-2 runs |
| `runs/garg_lr/` | the learning-rate check (Appendix B) |
| `runs/linattn.json`, `runs/linattn_rotated.json` | the linear-attention runs (inputs aligned with the axes, and rotated) |
| `jobs.txt` | the command for every GPT-2 run |

The learning rates of the GPT-2 runs come from sweeps on a smaller GPT-2 in an earlier pilot; that pilot's code, logs
and report are in the git history (commit `2ee09b1`).
