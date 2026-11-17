"""Seed-level summary of the Garg skewed-covariance test from a garg_eval JSON.  python scripts/skewed_summary.py file.json"""
import json, sys
import numpy as np
R = json.load(open(sys.argv[1]))
ks = [7, 8, 10, 16, 32]
groups = {}
for n, meta in R["models"].items():
    groups.setdefault(meta["opt"], []).append(n)
for s in ("standard", "skewed", "half_subspace", "scale_x2", "noisy"):
    print(f"[{s}] relative error at k =", ks, "  | rises after k=7? (err@32 > err@7)")
    for opt, names in groups.items():
        for n in names:
            v = R[s][n]
            print(f"   {opt:6s} {n:28s}", " ".join(f"{v[k]:.3f}" for k in ks), "  rises" if v[32] > v[7] else "")
        m = np.array([[R[s][n][k] for k in ks] for n in names])
        print(f"   {opt:6s} {'mean over ' + str(len(names)) + ' models':28s}", " ".join(f"{x:.3f}" for x in m.mean(0)))
    print("   least squares                       ", " ".join(f"{R[s]['least squares'][k]:.3f}" for k in ks))
