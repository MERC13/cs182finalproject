"""Export every number shown on the results dashboard from the run logs, as CSV files.

  python scripts/export_dashboard_data.py figures/dashboard

Each file is one dataset of the dashboard. Nothing here is typed in by hand except the condensed
text of the reproduction and hypothesis tables; every number is computed from runs/ and the saved
analysis outputs (figures/confirm/confirm.json, figures/transience/transience.json,
figures/noise/linattn_redesign.json).
"""
from __future__ import annotations

import csv
import glob
import json
import os
import re
import subprocess
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
from icl.analysis import load  # noqa: E402
from analyze_confirm import run_metrics  # noqa: E402
from analyze_linattn import metrics as la_metrics  # noqa: E402
from analyze_noise import beta  # noqa: E402
import analyze_transience as AT  # noqa: E402

OPT_NAME = {"sgd": "SGD", "gd": "SGD", "adam": "Adam", "adamw": "AdamW", "signgd": "SignGD", "muon": "Muon",
            "muon_exact": "Muon (exact)", "muon_m0": "Muon (no momentum)", "muon_exact_m0": "Muon (exact, no momentum)"}


def num(x, nd=4):
    return "" if x is None or (isinstance(x, float) and not np.isfinite(x)) else round(float(x), nd)


def write(out, name, cols, rows):
    with open(os.path.join(out, f"{name}.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c, "") for c in cols})
    print(f"{name}.csv: {len(rows)} rows")


def stats(v):
    v = np.array([x for x in v if x is not None and np.isfinite(x)], float)
    return (np.median(v), v.min(), v.max(), len(v)) if len(v) else (np.nan, np.nan, np.nan, 0)


# ---------------------------------------------------------------- facts (prose numbers)
def facts():
    C = json.load(open(os.path.join(ROOT, "figures/confirm/confirm.json")))
    med = lambda key, k: float(np.nanmean([r[k] for r in C[key]]))  # mean over the seeds that got there, as in the report
    rows = []
    add = lambda key, value, fmt, label, source: rows.append(dict(key=key, value=num(value), fmt=fmt, label=label, source=source))
    src = "runs/confirm (seeds 10-12), scripts/analyze_confirm.py"
    for cond, tag in (("iso", "iso"), ("1/i2-ali", "ani"), ("1/i2-rot", "rot")):
        a, m = med(f"{cond}|adamw", "T_conv"), med(f"{cond}|muon", "T_conv")
        add(f"conv_{tag}_adamw", a, "int", f"AdamW steps to converge, {cond} (mean over seeds that converged)", src)
        add(f"conv_{tag}_muon", m, "int", f"Muon steps to converge, {cond} (mean over seeds that converged)", src)
        add(f"speedup_{tag}", a / m, "x1", f"Muon convergence speed-up over AdamW, {cond}", src)
    for cond, tag in (("iso", "iso"), ("1/i2-ali", "ani")):
        for o in ("adamw", "muon"):
            add(f"weak_{tag}_{o}", med(f"{cond}|{o}", "T_weak"), "int", f"{o} step at which the weakest axis is learned, {cond}", src)
    for i, lv in enumerate((0.03, 0.01)):
        qa = np.nanmedian([r["q1_matched"][i] for r in C["1/i2-ali|adamw"]])
        qm = np.nanmedian([r["q1_matched"][i] for r in C["1/i2-ali|muon"]])
        add(f"h3_ratio_{str(lv).replace('.', '')}", qa / qm, "x1", f"AdamW / Muon shifted-query error at matched in-distribution error {lv}", src)
    T = json.load(open(os.path.join(ROOT, "figures/transience/transience.json")))
    for o in ("adamw", "muon"):
        fades = [r["fade"] for r in T["runs"] if r["M"] == 32 and r["opt"] == o and r["fade"]]
        add(f"fade_m32_{o}", np.median(fades), "int", f"{o}: step at which ICL fades with 32 training tasks (median of 3 seeds)",
            "runs/transience, runs/transience_ext, scripts/analyze_transience.py")
    B = {}
    for f in glob.glob(os.path.join(ROOT, "runs/init/*_init044_s1[01].json")):
        if "lr0.3" in f:
            continue
        r = load(f)
        o = "sgd" if "sgd" in os.path.basename(f) else r.cfg["opt"]
        B.setdefault(o, []).append(beta(r.axis_learn_steps(0.5), r.lam))
    for o in ("sgd", "adamw", "muon"):
        add(f"beta044_{o}", np.mean(B[o]), "f2", f"{o} spectral learning exponent, GPT-2 with concatenated tokens and init 0.044 (mean of 2 seeds)",
            "runs/init (seeds 10, 11), scripts/analyze_init.py")
    D = json.load(open(os.path.join(ROOT, "figures/noise/linattn_redesign.json")))
    for cond, tag in (("1/i2-ali", "i2"), ("1/i-ali", "i")):
        a = np.nanmedian([q[1] for q in D[f"{cond}|adam|1024"]["q"]])
        m = np.nanmedian([q[1] for q in D[f"{cond}|muon_exact|1024"]["q"]])
        add(f"ood_ratio_{tag}", a / m, "x1", f"Adam / exact-Muon excess shifted-query loss at matched in-distribution error 0.05, {cond}, batch 1,024",
            "runs/linattn/noise_sweep*.json, scripts/analyze_linattn_redesign.py")
    return rows


# ---------------------------------------------------------------- text tables
REPRO = [
    ("zhang", "Zhang et al. 2025", "Linear attention with separate key and query weights learns principal axes one at a time, on predicted loss plateaus",
     "All 16 predicted plateaus matched to 5-6 decimals; axes switch on in eigenvalue order", "Yes, exactly"),
    ("vonoswald", "von Oswald et al. 2023", "A trained linear self-attention layer implements one preconditioned gradient-descent step",
     "The converged weights equal the analytic one-step solution to about 1e-14", "Yes, exactly"),
    ("ma", "Ma et al. 2026", "Simplified Muon converges at a rate independent of the condition number; gradient descent does not",
     "Gradient descent slows with the condition number; Muon does not, aligned or rotated", "Yes"),
    ("deeplinear", "Saxe et al. / Dragutinović et al. 2026", "In a deep linear network gradient descent learns modes one at a time; spectral descent learns them together",
     "Gradient descent sequential, spectral descent parallel, both unchanged by rotation", "Yes"),
    ("garg", "Garg et al. 2022", "A GPT-2 matches least squares in context and fails on a skewed test covariance",
     "Matches least squares in distribution; the skewed-covariance failure appears for an AdamW-trained model", "Yes"),
    ("xie", "Xie et al. 2025", "Rotating the whole parameter space slows Adam sharply",
     "AdamW in a rotated parameter space never leaves the plateau in our GPT-2", "Yes, strongly"),
    ("gargcur", "Garg et al. 2022, training recipe", "Their curriculum speeds up training",
     "Faster early training; the size of the skewed-covariance failure depends on training details", "Yes, with a nuance"),
]
HYP = [
    ("H1", "Muon reaches low error on unseen tasks sooner", "Confirmed: faster to converge on every input covariance tested", "Confirmed"),
    ("H2", "Skewed inputs slow SGD more than Muon", "Confirmed for AdamW (the weakest direction arrives much later); SGD could not be trained in the original GPT-2", "Confirmed for AdamW"),
    ("H3", "Muon handles queries on low-variance directions better", "Confirmed: lower shifted-query error at matched in-distribution error", "Confirmed"),
    ("H4", "SGD runs PCR on a growing set of directions, Muon a ridge-like estimator", "Revised: the staircase belongs to AdamW; Muon learns in the same order, compressed", "Revised"),
    ("Rot", "Rotating the input covariance changes Adam's learning", "No effect in the GPT-2 at batch 64 or 256, even with an identity read-in; large in one-layer linear attention", "Null in the GPT-2, dropped"),
    ("SGD", "SGD can be compared with AdamW and Muon in the GPT-2", "Never trained in the original GPT-2; trains with concatenated tokens and a larger hidden initialisation", "Fixed by redesign"),
    ("Trans", "Muon is the most transient (follow-up)", "Not supported: both follow the same path from ridge to memorisation; Muon is faster", "Not supported"),
]


# ---------------------------------------------------------------- learning order in the GPT-2
GPT2_RUNS = [("original", "runs/confirm/{o}_1i2-{c}_s{s}.json", ("adamw", "muon"), ("ali", "rot"), (10, 11, 12)),
             ("concat_init0.02", "runs/init/{o}_init002_s{s}.json", ("adamw", "muon"), ("ali",), (10,)),
             ("concat_init0.044", "runs/init/{o}_init044_s{s}.json", ("sgd_lr0.1", "adamw", "muon"), ("ali",), (10, 11)),
             ("concat_init0.088", "runs/rescue/{o}_1i2-{c}_s{s}.json", ("sgd", "adamw", "muon"), ("ali",), (10, 11))]
SETTING = {"original": "Original GPT-2", "concat_init0.02": "Concatenated tokens, init 0.02",
           "concat_init0.044": "Concatenated tokens, init 0.044", "concat_init0.088": "Concatenated tokens, init 0.088"}


def gpt2_runs():
    for setting, pat, opts, conds, seeds in GPT2_RUNS:
        for o in opts:
            for c in conds:
                for s in seeds:
                    f = os.path.join(ROOT, pat.format(o=o, c=c, s=s))
                    if os.path.exists(f):
                        yield setting, ("sgd" if o.startswith("sgd") else o), c, s, f


def axis_tables():
    ax, br, prof = [], [], []
    for setting, o, c, s, f in gpt2_runs():
        r = load(f)
        T = r.axis_learn_steps(0.5)
        m = run_metrics(r)
        for j, t in enumerate(T):
            ax.append(dict(run=f"{setting}|{o}|{c}|{s}", setting=SETTING[setting], opt=OPT_NAME[o], cond="aligned" if c == "ali" else "rotated",
                           seed=s, direction=j + 1, step=num(t, 0)))
        br.append(dict(run=f"{setting}|{o}|{c}|{s}", setting=SETTING[setting], opt=OPT_NAME[o], cond="aligned" if c == "ali" else "rotated",
                       seed=s, beta=num(beta(T, r.lam), 3), stair=num(T[-1] / T[0] if np.isfinite(T[-1]) else np.nan, 1),
                       converged=num(m["T_conv"], 0), final_error=num(float(r.err_mean[-1]), 4)))
        if c == "ali":
            i = np.nonzero(r.err_last <= 0.03)[0]
            if len(i):
                p = 1 - r.probe_err()[int(i[0])]
                for j, v in enumerate(p):
                    prof.append(dict(run=f"{setting}|{o}|{s}|{j + 1}", setting=SETTING[setting], opt=OPT_NAME[o], seed=s, direction=j + 1,
                                     learned=num(v, 3), step=int(r.steps[int(i[0])])))
    return ax, br, prof


# ---------------------------------------------------------------- why SGD could not train
def sgd_blocks():
    out = subprocess.run([sys.executable, os.path.join(ROOT, "scripts/grad_blocks.py")], capture_output=True, text=True, cwd="/tmp").stdout
    share, rms, change = {}, {}, {}
    for line in out.splitlines():
        p = line.split()
        if len(p) == 5 and p[0].endswith(".weight") and re.match(r"^\d+$", p[1]):
            share[p[0]], rms[p[0]] = float(p[3]), float(p[4])
        elif len(p) == 5 and p[0].endswith(".weight"):
            try:
                change[p[0]] = [float(x) for x in p[2:]]
            except ValueError:
                pass
    names = {"read_out.weight": "Read-out", "read_in.weight": "Read-in", "q.weight": "Query", "k.weight": "Key", "v.weight": "Value",
             "o.weight": "Attention output", "fc.weight": "MLP input", "proj.weight": "MLP output"}
    rows = []
    for k, n in names.items():
        if k in change:
            rows.append(dict(block=n, share=num(share.get(k, np.nan), 8), rms_grad=num(rms.get(k, np.nan), 8),
                             sgd_change=num(change[k][0], 6), adamw_change=num(change[k][1], 6), muon_change=num(change[k][2], 6)))
    return rows


def sgd_rescue():
    cfgs = [("Separate tokens, standard init", "SGD", "runs/sweep/sgd_lr0.1_clip1_iso.json"),
            ("Concatenated tokens, standard init", "SGD", "runs/concat/sgd_lr0.1_iso_s0.json"),
            ("Separate tokens, init 0.088", "SGD", "runs/rescue/sgd_sep_init088_iso_s0.json"),
            ("Concatenated tokens, init 0.088", "SGD", "runs/concat/sgd_lr0.1_init088_iso_s0.json"),
            ("Concatenated tokens, init 0.088", "AdamW", "runs/concat/adamw_init088_iso_s0.json")]
    rows = []
    for name, o, f in cfgs:
        L = json.load(open(os.path.join(ROOT, f)))
        for e in L["evals"]:
            if "test_err_k" in e and e["step"] <= 3000:
                rows.append(dict(run=f"{name}|{o}|{e['step']}", config=name, opt=o, step=e["step"], error=num(np.mean(e["test_err_k"][1:]), 4)))
    return rows


# ---------------------------------------------------------------- rotation in the one-layer model
def rotation():
    R = [r for r in json.load(open(os.path.join(ROOT, "runs/linattn/noise_sweep.json"))) if r.get("selected")]
    rows = []
    for o in ("adam", "signgd", "muon"):
        for b in (16, 64, 256, 1024):
            ta = [la_metrics(r)["T_conv"] for r in R if r["opt"] == o and r["batch"] == b and r["cond"] == "1/i2-ali"]
            tr = [la_metrics(r)["T_conv"] for r in R if r["opt"] == o and r["batch"] == b and r["cond"] == "1/i2-rot"]
            a = np.median(ta) if np.all(np.isfinite(ta)) else np.nan
            t = np.median(tr) if np.all(np.isfinite(tr)) else np.nan
            rows.append(dict(run=f"{o}|{b}", opt=OPT_NAME[o], batch=str(b), t_aligned=num(a, 0), t_rotated=num(t, 0),
                             cost=num(t / a if np.isfinite(a) and np.isfinite(t) else np.nan, 2)))
    E = json.load(open(os.path.join(ROOT, "runs/linattn/exact_sweep.json")))
    for o in ("adam", "signgd", "muon"):
        best = {}
        for c in ("1/i2-ali", "1/i2-rot"):
            v = [e["T_conv"] for e in E if e["opt"] == o and e["mode"] == "sep" and e["cond"] == c and abs(e["init"] - 0.01) < 1e-9
                 and e["T_conv"] is not None and np.isfinite(e["T_conv"])]
            best[c] = min(v) if v else np.nan
        rows.append(dict(run=f"{o}|exact", opt=OPT_NAME[o], batch="exact", t_aligned=num(best["1/i2-ali"], 0), t_rotated=num(best["1/i2-rot"], 0),
                         cost=num(best["1/i2-rot"] / best["1/i2-ali"], 2)))
    return rows


# ---------------------------------------------------------------- transience
def transience():
    runs = AT.load_runs([os.path.join(ROOT, "runs/transience/*.json"), os.path.join(ROOT, "runs/transience_ext/*.json")])
    curves = []
    for R in runs:
        c = R["c"]
        for st, s, m in zip(c["step"], c["s_sm"], c["m_sm"]):
            curves.append(dict(run=f"{R['opt']}|{R['M']}|{R['seed']}|{int(st)}", M=R["M"], opt=OPT_NAME[R["opt"]], seed=R["seed"],
                               step=int(st), icl=num(s, 4), memorisation=num(m, 4)))
    T = json.load(open(os.path.join(ROOT, "figures/transience/transience.json")))
    timing = [dict(run=f"{r['opt']}|{r['M']}|{r['seed']}", M=r["M"], opt=OPT_NAME[r["opt"]], seed=r["seed"], steps=r["last"],
                   emerge=num(r["emerge"], 0), peak_icl=num(r["s_peak"], 3), fade=num(r["fade"], 0), memorised=num(r["memorised"], 0),
                   fade_over_emerge=num(r["fade_over_emerge"], 2))
              for r in T["runs"] if r["opt"] in ("adamw", "muon")]
    matched = []
    for key, v in T["matched"].items():
        M, lv = re.match(r"M(\d+) m=(-?[\d.]+)", key).groups()
        for o in ("adamw", "muon"):
            md, lo, hi, n = stats(v[o])
            matched.append(dict(run=f"{M}|{lv}|{o}", M=int(M), m_level=float(lv), opt=OPT_NAME[o], mean=num(np.mean([x for x in v[o] if x is not None]), 3),
                                lo=num(lo, 3), hi=num(hi, 3), n=n))
    return curves, timing, matched


# ---------------------------------------------------------------- noise, shapes and out-of-distribution error (one-layer model)
LA_FILES = {"1/i2-ali": ["runs/linattn/noise_sweep.json", "runs/linattn/noise_sweep_muonvar_exact.json", "runs/linattn/noise_sweep_muonvar_m0.json"],
            "1/i-ali": ["runs/linattn/noise_sweep_1i.json"]}
SPEC = {"1/i2-ali": "λ ∝ 1/i²", "1/i-ali": "λ ∝ 1/i"}


def la_runs(cond):
    out = []
    for f in LA_FILES[cond]:
        for r in json.load(open(os.path.join(ROOT, f))):
            if r.get("selected") and r["cond"] == cond:
                o = r["opt"] + ("_m0" if r.get("muon_momentum") == 0 else "")
                out.append((o, r))
    return out


def rho_at(m, lv):
    exc = m["exc"]
    i = np.nonzero(exc <= lv)[0]
    if not len(i):
        return None
    i = int(i[0])
    if i == 0:
        return m["rho"][0]
    t = (exc[i - 1] - lv) / (exc[i - 1] - exc[i])
    return m["rho"][i - 1] + t * (m["rho"][i] - m["rho"][i - 1])


def one_layer():
    betas, gains = [], []
    for cond in LA_FILES:
        R = la_runs(cond)
        for o in sorted({o for o, _ in R}):
            for b in (16, 64, 256, 1024):
                v = [(r, la_metrics(r)) for oo, r in R if oo == o and r["batch"] == b]
                if not v:
                    continue
                md, lo, hi, n = stats([beta(m["T_axis"], r["lam"]) for r, m in v])
                betas.append(dict(run=f"{cond}|{o}|{b}", model=f"One-layer linear attention, {SPEC[cond]}", opt=OPT_NAME[o], batch=b,
                                  beta=num(md, 3), lo=num(lo, 3), hi=num(hi, 3), n=n))
                if o in ("gd", "adam", "muon", "muon_exact"):
                    P = [rho_at(m, 0.1) for _, m in v]
                    P = [p for p in P if p is not None]
                    if P:
                        p = np.median(P, 0)
                        for j, g in enumerate(p):
                            gains.append(dict(run=f"{cond}|{o}|{b}|{j + 1}", spectrum=SPEC[cond], opt=OPT_NAME[o], batch=b, direction=j + 1, gain=num(g, 3)))
    # GPT-2 (original layout), Muon, aligned 1/i^2
    g = {16: ["runs/noise/muon_b16_1i2-ali_s10.json"], 64: [f"runs/confirm/muon_1i2-ali_s{s}.json" for s in (10, 11, 12)],
         256: ["runs/noise/muon_b256_1i2-ali_s10.json"]}
    for b, fs in g.items():
        v = []
        for f in fs:
            r = load(os.path.join(ROOT, f))
            v.append(beta(r.axis_learn_steps(0.5), r.lam))
        md, lo, hi, n = stats(v)
        betas.append(dict(run=f"gpt2|muon|{b}", model="GPT-2, original layout, λ ∝ 1/i²", opt="Muon", batch=b, beta=num(md, 3), lo=num(lo, 3), hi=num(hi, 3), n=n))
    D = json.load(open(os.path.join(ROOT, "figures/noise/linattn_redesign.json")))
    ood = []
    for key, v in D.items():
        cond, o, b = key.split("|")
        for i, lv in enumerate((0.1, 0.05)):
            md, lo, hi, n = stats([q[i] for q in v["q"]])
            ood.append(dict(run=f"{cond}|{o}|{b}|{lv}", spectrum=SPEC[cond], opt=OPT_NAME[o], batch=int(b), level=lv,
                            excess=num(md, 4), lo=num(lo, 4), hi=num(hi, 4), n=n))
    return betas, gains, ood


def main():
    out = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "figures/dashboard")
    os.makedirs(out, exist_ok=True)
    write(out, "facts", ["key", "value", "fmt", "label", "source"], facts())
    write(out, "reproductions", ["id", "paper", "claim", "result", "verdict"],
          [dict(zip(["id", "paper", "claim", "result", "verdict"], r)) for r in REPRO])
    write(out, "hypotheses", ["id", "hypothesis", "finding", "status"], [dict(zip(["id", "hypothesis", "finding", "status"], r)) for r in HYP])
    ax, br, prof = axis_tables()
    write(out, "axis_times", ["run", "setting", "opt", "cond", "seed", "direction", "step"], ax)
    write(out, "beta_runs", ["run", "setting", "opt", "cond", "seed", "beta", "stair", "converged", "final_error"], br)
    write(out, "gpt2_profiles", ["run", "setting", "opt", "seed", "direction", "learned", "step"], prof)
    write(out, "sgd_blocks", ["block", "share", "rms_grad", "sgd_change", "adamw_change", "muon_change"], sgd_blocks())
    write(out, "sgd_rescue", ["run", "config", "opt", "step", "error"], sgd_rescue())
    write(out, "rotation_testbed", ["run", "opt", "batch", "t_aligned", "t_rotated", "cost"], rotation())
    curves, timing, matched = transience()
    write(out, "transience_curves", ["run", "M", "opt", "seed", "step", "icl", "memorisation"], curves)
    write(out, "transience_timing", ["run", "M", "opt", "seed", "steps", "emerge", "peak_icl", "fade", "memorised", "fade_over_emerge"], timing)
    write(out, "transience_matched", ["run", "M", "m_level", "opt", "mean", "lo", "hi", "n"], matched)
    betas, gains, ood = one_layer()
    write(out, "beta_noise", ["run", "model", "opt", "batch", "beta", "lo", "hi", "n"], betas)
    write(out, "gain_profiles", ["run", "spectrum", "opt", "batch", "direction", "gain"], gains)
    write(out, "ood_matched", ["run", "spectrum", "opt", "batch", "level", "excess", "lo", "hi", "n"], ood)


if __name__ == "__main__":
    main()
