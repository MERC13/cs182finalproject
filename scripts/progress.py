"""One-line progress per run: step:err(mean k=1..32) samples.  python scripts/progress.py 'runs/x/*.json' [every]"""
import glob, json, sys
import numpy as np
every = int(sys.argv[2]) if len(sys.argv) > 2 else 500
for f in sorted(glob.glob(sys.argv[1])):
    try:
        L = json.load(open(f))
    except Exception:
        continue
    E = L["evals"]
    s = " ".join(f"{e['step']}:{np.mean(e['test_err_k'][1:]):.3f}" for e in E if e["step"] % every == 0)
    print(f"{f.split('/')[-1]:34s} {L['status']:8s} val={L.get('final_val_loss')} | {s}")
