"""Learning curves of the iso LR sweep, one panel per optimizer.  python scripts/plot_sweep.py 'runs/sweep/*.json' out.png"""
import glob, json, os, sys
from collections import defaultdict
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from plot_pilot import plt, logy, INK2  # noqa: E402
RAMP = ["#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#104281", "#0d366b"]  # light = small lr
runs = defaultdict(list)
files = [f for pat in sys.argv[1].split(",") for f in sorted(glob.glob(pat))]
for f in files:
    L = json.load(open(f)); c = L["config"]
    if c.get("cond") != "iso" or c.get("y_mode") != "sep" or c.get("d", 8) != 8:
        continue
    tag = c["opt"]
    lab = f"lr {c['lr']:g}" + (" (clip 1)" if c.get("clip") else "")
    if tag == "muon" and abs(c.get("aux_lr", 1e-3) - 1e-3) > 1e-9:
        lab += f" (aux {c['aux_lr']:g})"
    E = [e for e in L["evals"] if e["step"] <= 4000]
    st = [e["step"] for e in E]; err = [np.mean(e["test_err_k"][1:]) for e in E]
    if L["status"] == "running" and E[-1]["step"] < 4000:
        lab += " (stopped)"
    runs[tag].append((c["lr"] + (1e-9 if c.get("clip") else 0), lab, st, err, L["status"]))
names = {"sgd": "SGD + momentum 0.9", "adamw": "AdamW", "muon": "Muon (aux AdamW 1e-3 unless noted)"}
fig, axes = plt.subplots(1, 3, figsize=(11, 3.9), sharey=True)
for ax, opt in zip(axes, ["sgd", "adamw", "muon"]):
    rs = sorted(runs[opt])
    for i, (lr, lab, st, err, status) in enumerate(rs):
        col = RAMP[int(round(i * (len(RAMP) - 1) / max(1, len(rs) - 1)))]
        if status.startswith("diverged"):
            lab += " — diverged"
        ax.plot(st, err, color=col, lw=1.8, label=lab)
    logy(ax, 0.1, 1.2); ax.set_title(names[opt]); ax.set_xlabel("training step")
    ax.legend(fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2)
axes[0].set_ylabel("test error, mean over k = 1..32\n(1 = predict 0; LS floor 0.117)")
for ax in axes:
    ax.axhline(0.1168, color=INK2, lw=0.8)
fig.suptitle("Isotropic inputs: learning-rate sweep (1 seed, 4k steps)", fontsize=10, fontweight="bold")
fig.tight_layout(); fig.savefig(sys.argv[2], dpi=150)
print("wrote", sys.argv[2])
