"""Load exported checkpoints and re-run their final evaluation; compare with the training log.

  python scripts/eval_checkpoint.py checkpoints/garg_scale/muon_s11.pt [more .pt files ...]
  python scripts/eval_checkpoint.py --all

For each checkpoint: rebuild the model from its stored config, load the weights, rebuild the same evaluation prompts
(the run's --eval_seed), and recompute the in-distribution error by number of examples, the per-direction probe
errors and the shifted-query errors. These are compared with the last evaluation in the run's log (the checkpoint
holds exactly the weights evaluated there). Also prints the final validation loss on the separate validation prompts
used for learning-rate selection (seed 10,000).
"""
import argparse
import glob
import json
import os
import sys
from types import SimpleNamespace

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from icl.data import make_condition, make_eval_sets  # noqa: E402
from icl.estimators import Reference  # noqa: E402
from icl.model import GPT2ICL  # noqa: E402
from icl.train import VAL_SEED, evaluate, predict  # noqa: E402


def load_model(path):
    ck = torch.load(path, weights_only=False, map_location="cpu")
    # flags added after a run was made are absent from its config: fill in the defaults that run used
    a = SimpleNamespace(**{"y_mode": "garg", "readin_init": "torch", "hidden_init": 0.0, "qshift_ts": None,
                           "n_test": 1000, **ck["config"]})
    model = GPT2ICL(a.d, a.n_points, a.n_embd, a.n_layer, a.n_head, a.y_mode, a.readin_init, a.hidden_init)
    model.load_state_dict(ck["model"])
    model.eval()
    return model, a, ck


def check(path):
    model, a, ck = load_model(path)
    cond = make_condition(a.cond, a.d)
    kw = {"qts": tuple(float(t) for t in a.qshift_ts.split(","))} if a.qshift_ts else {}
    ev = make_eval_sets(cond, a.n_points, a.eval_seed, n_test=a.n_test, **kw)
    ref = Reference(ev.x, ev.y, cond.lam, cond.U)
    out = evaluate(model, ev, ref, cond, full=True, qshift=True)
    log = json.load(open(os.path.join(ROOT, ck["log"])))
    last = log["evals"][-1]
    assert last["step"] == ck["step"], (last["step"], ck["step"])
    diffs = {}
    for key in ("test_err_k", "probe_err_k", "qshift_err"):
        if key in last:
            mine, theirs = np.array(out[key], dtype=float), np.array(last[key], dtype=float)
            diffs[key] = float(np.max(np.abs(mine - theirs) / np.maximum(np.abs(theirs), 1e-3)))
    evv = make_eval_sets(cond, a.n_points, VAL_SEED, n_test=a.n_test, n_probe=1, n_qshift=1)
    pv = predict(model, evv.x, evv.y)
    val = float((((pv - evv.y) ** 2).mean(0) / a.d).mean())
    worst = max(diffs.values())
    print(f"{os.path.relpath(path, ROOT):55s} step {ck['step']:5d}  error@{a.n_points - 1} {out['test_err_k'][-1]:.4f}"
          f"  max rel. diff vs log {worst:.1e}  val loss {val:.4f} (log {log.get('final_val_loss')})")
    return worst


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="*")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--tol", type=float, default=1e-3)
    a = ap.parse_args()
    torch.set_num_threads(max(1, os.cpu_count() or 1))
    paths = sorted(glob.glob(os.path.join(ROOT, "checkpoints", "*", "*.pt"))) if a.all else a.paths
    bad = [p for p in paths if check(p) > a.tol]
    print(f"{len(paths) - len(bad)}/{len(paths)} checkpoints reproduce their logged final evaluation (tolerance {a.tol})")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
