"""Rank sweep runs per optimizer by area under the test-error curve (mean over evals of err k=1..32)."""
import glob, json, sys
from collections import defaultdict
import numpy as np
rows = defaultdict(list)
for f in sorted(glob.glob(sys.argv[1])):
    L = json.load(open(f)); c = L["config"]
    E = [e for e in L["evals"] if e["step"] <= int(sys.argv[2]) if len(sys.argv) > 2] if len(sys.argv) > 2 else L["evals"]
    err = np.array([np.mean(e["test_err_k"][1:]) for e in E])
    esc = next((e["step"] for e in E if np.mean(e["test_err_k"][1:]) < 0.9), None)
    tag = c["opt"] + ("/clip" if c.get("clip") else "")
    rows[tag].append((float(err.mean()), c["lr"], esc, float(err[-1]), E[-1]["step"], L["status"]))
for opt, rs in rows.items():
    print(opt)
    for auc, lr, esc, fin, st, status in sorted(rs):
        print(f"   lr={lr:<8g} AUC={auc:.3f}  escape(<0.9)@{esc}  err@{st}={fin:.3f}  {status}")
