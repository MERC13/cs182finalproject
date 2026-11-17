"""bf16-autocast vs fp32 training, same seed/config: max and final difference of the test-error curve."""
import json
import numpy as np
pairs = [("runs/sweep/adamw_lr3e-4_iso.json", "runs/search/adamw_lr3e-4_b20.999_iso_fp32_s0.json", "AdamW 3e-4 iso"),
         ("runs/sweep/muon_lr0.005_iso.json", "runs/search/muon_lr0.005_iso_fp32_s0.json", "Muon 0.005 iso")]
for bf, fp, name in pairs:
    A, B = json.load(open(bf)), json.load(open(fp))
    ea = {e["step"]: np.mean(e["test_err_k"][1:]) for e in A["evals"]}
    eb = {e["step"]: np.mean(e["test_err_k"][1:]) for e in B["evals"]}
    common = sorted(set(ea) & set(eb))
    d = [abs(ea[s] - eb[s]) for s in common]
    esc = lambda e: next((s for s in sorted(e) if e[s] < 0.9), None)
    print(f"{name}: steps compared {common[0]}..{common[-1]} | max |Δerr| {max(d):.3f} | final bf16 {ea[common[-1]]:.3f} "
          f"vs fp32 {eb[common[-1]]:.3f} | escape bf16 {esc(ea)} vs fp32 {esc(eb)} | fp32 status {B['status']}")
