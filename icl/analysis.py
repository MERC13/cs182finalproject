"""Load run logs and compute the pilot's summary metrics."""
from __future__ import annotations

import glob
import json
from dataclasses import dataclass

import numpy as np

OPT_ORDER = ["sgd", "adamw", "muon"]
OPT_LABEL = {"sgd": "SGD+mom", "adamw": "AdamW", "muon": "Muon"}
# categorical slots 1-3 of the validated reference palette (all-pairs safe)
OPT_COLOR = {"sgd": "#2a78d6", "adamw": "#eb6834", "muon": "#1baf7a"}
COND_ORDER = ["iso", "1/i-ali", "1/i-rot", "1/i2-ali", "1/i2-rot"]
COND_LABEL = {"iso": "isotropic", "1/i-ali": "λ∝1/i, aligned", "1/i-rot": "λ∝1/i, rotated",
              "1/i2-ali": "λ∝1/i², aligned", "1/i2-rot": "λ∝1/i², rotated"}


@dataclass
class Run:
    path: str
    cfg: dict
    status: str
    lam: np.ndarray
    steps: np.ndarray
    err_k: np.ndarray  # (S, n) normalised test error per prefix
    probe: np.ndarray | None  # (S, d, n)
    dist: dict | None  # name -> (S,) distance averaged over k = 1..32
    dist8: dict | None
    q_steps: np.ndarray
    qshift: np.ndarray  # (Sq, T)
    iw_gain: np.ndarray  # (Sq, d) implied-weight gain per principal axis (1 = least squares)
    iw_err: np.ndarray
    iw_r2: np.ndarray
    ref_err: dict  # estimator name -> (n,)
    final_val: float | None
    wall: float | None

    @property
    def opt(self):
        return self.cfg["opt"]

    @property
    def cond(self):
        return self.cfg["cond"]

    @property
    def lr(self):
        return self.cfg["lr"]

    @property
    def err_mean(self):  # mean over k = 1..32
        return self.err_k[:, 1:].mean(1)

    @property
    def err_last(self):  # 32 in-context examples
        return self.err_k[:, -1]

    def excess_mean(self):
        return self.err_mean - self.ref_err["ols"][1:].mean()

    def first_below(self, series, thr):
        idx = np.nonzero(series < thr)[0]
        return float(self.steps[idx[0]]) if len(idx) else np.nan

    def read_at(self, values, lv, interp=True):
        """A per-evaluation quantity (first axis aligned with self.steps) read at matched error lv.
        interp=True: at exactly lv, interpolating linearly in log error between the last evaluation above lv and the
        first at or below it (the paper's H1 reads). interp=False: at the first evaluation at or below lv.
        None if the run never reaches lv."""
        e = self.err_last
        idx = np.nonzero(e <= lv)[0]
        if not len(idx):
            return None
        i = int(idx[0])
        v = np.asarray(values)
        if not interp or i == 0:
            return v[i]
        w = (np.log(e[i - 1]) - np.log(lv)) / (np.log(e[i - 1]) - np.log(e[i]))
        return v[i - 1] + w * (v[i] - v[i - 1])

    def probe_err(self):
        """(S, d) probe relative error per principal axis, averaged over k = 16..32."""
        return self.probe[:, :, 16:].mean(2)

    def axis_learn_steps(self, thr=0.5):
        """Step at which each principal axis' probe error first drops below thr."""
        P = self.probe_err()
        return np.array([self.first_below(P[:, j], thr) for j in range(P.shape[1])])


def load(path) -> Run:
    L = json.load(open(path))
    E = L["evals"]
    steps = np.array([e["step"] for e in E])
    err_k = np.array([e["test_err_k"] for e in E])
    has_full = "probe_err_k" in E[-1]
    probe = np.array([e["probe_err_k"] for e in E]) if has_full else None
    dist = {k: np.array([e["dist_1_32"][k] for e in E]) for k in E[-1]["dist_1_32"]} if has_full else None
    dist8 = {k: np.array([e["dist_8_32"][k] for e in E]) for k in E[-1]["dist_8_32"]} if has_full else None
    qe = [e for e in E if "qshift_err" in e]
    ref_err = dict(zip(L["ref_names"], np.array(L["ref_err_k"])))
    ie = [e for e in qe if "iw_gain" in e]
    return Run(path, L["config"], L["status"], np.array(L["cond"]["lam"]), steps, err_k, probe, dist, dist8,
               np.array([e["step"] for e in qe]), np.array([e["qshift_err"] for e in qe]),
               np.array([e["iw_gain"] for e in ie]), np.array([e["iw_err"] for e in ie]),
               np.array([e["iw_r2"] for e in ie]), ref_err, L.get("final_val_loss"), L.get("wall"))


def load_all(pattern) -> list[Run]:
    out = []
    for p in sorted(glob.glob(pattern)):
        try:
            out.append(load(p))
        except Exception as e:  # partial file
            print("skip", p, e)
    return out


def best_fit(run: Run, prefix: str, which: str = "dist"):
    """Per eval step: (best estimator suffix, its distance) within a family (ridge_/pcr_/gdpre_)."""
    D = getattr(run, which)
    names = [k for k in D if k.startswith(prefix)]
    M = np.stack([D[k] for k in names])  # (F, S)
    i = M.argmin(0)
    return [names[j].split("_", 1)[1] for j in i], M.min(0)
