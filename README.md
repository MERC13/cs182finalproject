# How Optimizers Shape In-Context Learning

Code and data for the CS 182 project paper ([`paper/main.pdf`](paper/main.pdf)). We train a GPT-2 and one-layer
linear attention on in-context linear regression with SGD, AdamW and Muon, and compare the in-context algorithm each
one learns.

```bash
pip install -r requirements.txt
```

## Reproduce the paper

Every figure and number is computed from the committed logs in `runs/`; this takes a few seconds.

```bash
python scripts/figures.py        # paper/figures/*.pdf
python scripts/results.py        # every number in the text and tables
cd paper && latexmk -pdf main.tex
```

## Retrain

```bash
grep -v '^#' jobs.txt | xargs -P 4 -L 1 python -m icl.train   # 31 GPT-2 runs, 2-5 CPU hours each
python -m icl.linattn runs/linattn.json                       # linear attention, about 5 minutes on 4 cores
```

Training is deterministic on one CPU thread, so these commands rewrite the same logs; the exception is two runs that
originally used several threads (`garg_scale/muon_pow1_lr0.0005_s10`, `garg_lr/muon_pow1_lr0.007_s10`), whose reruns
match to rounding at first and then drift. The final weights of every GPT-2 run are next to its log:

```python
from icl.model import GPT2
model = GPT2.load("runs/garg_scale/muon_s10.pt")   # model(x, y) -> predictions
```

## Layout

| Path | Contents |
| --- | --- |
| `icl/data.py` | the task and the evaluation prompts (in-distribution, probe, shifted queries) |
| `icl/model.py` | the GPT-2 |
| `icl/optim.py` | Muon and the spectral knob |
| `icl/estimators.py` | reference estimators: ridge, PCR, one-step gradient descent |
| `icl/train.py` | GPT-2 training and evaluation |
| `icl/linattn.py` | linear attention: model, exact losses, learning-rate sweep |
| `icl/analysis.py` | reading the logs at a matched error |
| `scripts/` | `figures.py` and `results.py` |
| `runs/garg_scale/` | logs (`.json`) and final weights (`.pt`) of the 12-layer GPT-2 runs |
| `runs/garg_lr/` | the learning-rate check (Appendix B) |
| `runs/linattn.json` | the linear-attention runs |
| `jobs.txt` | the command for every GPT-2 run |

The learning rates of the GPT-2 runs come from sweeps on a smaller GPT-2 in an earlier pilot; that pilot's code, logs
and report are in the git history (commit `2ee09b1`).
