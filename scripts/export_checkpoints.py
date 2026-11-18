"""Export the final model weights of every GPT-2 run the report uses to checkpoints/ (committed to the repo).

  python scripts/export_checkpoints.py

Each runs/<dir>/<name>.ckpt (model + optimizer state, kept out of git) becomes checkpoints/<dir>/<name>.pt holding
{"model": float32 state_dict, "config": the run's training flags, "step": the final step, "log": the source log path}.
The weights are those evaluated at the run's last logged evaluation; scripts/eval_checkpoint.py re-runs that evaluation.
The linear-attention runs need no checkpoints: their full learned matrix at every logged step is in
runs/linattn/ood_gamma.json.
"""
import os

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUNS = {
    # Garg-scale GPT-2 (12 layers, 8 heads, width 256, d = 20): Figures 1-3, Tables 1 and B
    "garg_scale": [f"{a}_s{s}" for a in ("sgd", "adamw", "muon", "muon_pow1_lr0.0005") for s in (10, 11, 12)]
                  + ["muon_exact_s10"],
    # smaller GPT-2 (4 layers, 4 heads, width 128, d = 8): Table 1, middle column
    "init": [f"{a}_init044_s{s}" for a in ("sgd_lr0.1", "adamw", "muon") for s in (10, 11)],
    "knob": [f"muon_pow{p}_init044_s10" for p in ("1", "0.25", "0")],
    "ood": [f"{a}_s10" for a in ("sgd", "adamw", "muon", "muon_exact")],
}


def main():
    total = 0
    for d, names in RUNS.items():
        os.makedirs(os.path.join(ROOT, "checkpoints", d), exist_ok=True)
        for n in names:
            src = os.path.join(ROOT, "runs", d, n + ".ckpt")
            ck = torch.load(src, weights_only=False, map_location="cpu")
            out = os.path.join(ROOT, "checkpoints", d, n + ".pt")
            torch.save({"model": {k: v.float().contiguous() for k, v in ck["model"].items()},
                        "config": ck["log"]["config"], "step": int(ck["step"]), "log": f"runs/{d}/{n}.json"}, out)
            size = os.path.getsize(out)
            total += size
            print(f"{out[len(ROOT) + 1:]:55s} step {ck['step']:5d}  {size / 1e6:6.1f} MB")
    print(f"total {total / 1e6:.0f} MB")


if __name__ == "__main__":
    main()
