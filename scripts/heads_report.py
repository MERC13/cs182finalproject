"""Head alignment for saved grid models.  python scripts/heads_report.py 'runs/grid/*_s0.json' figures/grid"""
import glob
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from icl.analysis import COND_LABEL, OPT_LABEL  # noqa: E402
from icl.data import make_condition  # noqa: E402
from icl.heads import head_alignment, summarize  # noqa: E402
from icl.model import GPT2ICL  # noqa: E402

torch.set_num_threads(1)
rows = []
for f in sorted(glob.glob(sys.argv[1])):
    pt = f.replace(".json", ".pt")
    if not os.path.exists(pt):
        continue
    c = json.load(open(f))["config"]
    m = GPT2ICL(c["d"], c["n_points"], c["n_embd"], c["n_layer"], c["n_head"], c["y_mode"], c.get("readin_init", "torch"))
    m.load_state_dict(torch.load(pt))
    cond = make_condition(c["cond"], c["d"])
    S = summarize(head_alignment(m, cond, n_prompts=256))
    name = c["opt"] + ("+in" if c.get("muon_params") == "hidden+in" else "")
    print(f"\n{os.path.basename(f)}  [{name} {c['cond']}]")
    for r in S:
        print(f"  layer {r['layer']+1}: R2 q={r['r2_q']:.2f} k={r['r2_k']:.2f}  axis share "
              + " ".join(f"{v:.2f}" for v in r["axis_share"]) + f"   head diag-total " + " ".join(f"{v:.2f}" for v in r["per_head_diag_total"]))
    rows.append(dict(run=os.path.basename(f), opt=name, cond=c["cond"], layers=[
        dict(layer=r["layer"], r2_q=r["r2_q"], r2_k=r["r2_k"], axis_share=np.asarray(r["axis_share"]).tolist(),
             diag_total=r["per_head_diag_total"]) for r in S]))
os.makedirs(sys.argv[2], exist_ok=True)
json.dump(rows, open(os.path.join(sys.argv[2], "heads.json"), "w"), indent=1)
