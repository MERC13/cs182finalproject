"""Reading the logs: values at a matched error, probe scores, shifted-query errors, linear-attention profiles."""
import json
from functools import cache

import numpy as np

from .data import eval_prompts, spectrum
from .train import D, EVAL_SEED, N, SHIFTS

SEEDS = (10, 11, 12, 13, 14)
ARMS = {"SGD": "sgd_s{}", "AdamW": "adamw_s{}", "Muon": "muon_s{}", "Muon, p = 1": "muon_pow1_lr0.0005_s{}"}


class Run:
    """One GPT-2 log. Every array's first axis is aligned with `steps`."""

    def __init__(self, path):
        with open(path) as f:
            log = json.load(f)
        ev = log["evals"]
        self.steps = np.array([e["step"] for e in ev])
        self.err_k = np.array([e["test_err_k"] for e in ev])  # error after k examples, k = 0..40
        self.err = self.err_k[:, -1]  # after all 40 examples: the matched quantity
        self.train_loss = np.array([np.nan if e["train_loss"] is None else e["train_loss"] for e in ev])
        self.probe = 1 - np.array([e["probe_err_k"] for e in ev])[:, :, 16:].mean(2)  # f_j, predictions with 16-40
        self.dist = {k: np.array([e["dist_8_32"][k] for e in ev]) for k in ev[0]["dist_8_32"]}
        self.shift = [np.array(e["qshift_err"]) * shift_scale() for e in ev if "qshift_err" in e]

    def first_step(self, level):
        i = np.nonzero(self.err <= level)[0]
        return int(self.steps[i[0]]) if len(i) else None

    def at(self, values, level):
        """`values` at error `level`, interpolated linearly in log error; None if the run never gets there."""
        e, v = self.err, np.asarray(values)
        i = np.nonzero(e <= level)[0]
        if not len(i):
            return None
        i = int(i[0])
        if i == 0:
            return v[0]
        w = (np.log(e[i - 1]) - np.log(level)) / (np.log(e[i - 1]) - np.log(e[i]))
        return v[i - 1] + w * (v[i] - v[i - 1])

    def shift_at(self, level):
        """Error per query shift t (SHIFTS order) where the unshifted error on the same prompts crosses `level`,
        interpolated in log error."""
        i0 = SHIFTS.index(0.0)
        for a, b in zip(self.shift[:-1], self.shift[1:]):
            if a[i0] > level >= b[i0]:
                w = (np.log(a[i0]) - np.log(level)) / (np.log(a[i0]) - np.log(b[i0]))
                return np.exp(np.log(a) + w * (np.log(b) - np.log(a)))
        return None


@cache
def shift_scale():
    """The logs hold sum (yhat - y)^2 / sum y^2 on the shifted queries; this factor converts it to the paper's
    error (1/d) mean (yhat - y)^2."""
    y = eval_prompts(spectrum(D), N, EVAL_SEED, SHIFTS)["shift"]["y"][..., -1].double()
    return ((y ** 2).sum(1) / (y.shape[1] * D)).numpy()


@cache
def run(name, seed=None):
    """A main run by arm and seed, run("Muon", 10), or any log by its path in runs/, run("garg_lr/sgd_lr0.03_s10")."""
    path = f"garg_scale/{ARMS[name].format(seed)}" if seed is not None else name
    return Run(f"runs/{path}.json")


def nearest(dist, family):
    return min(v for k, v in dist.items() if k.startswith(family))


# --- linear attention ---------------------------------------------------------------------------------------------

def linattn(path="runs/linattn.json"):
    """Per run: excess loss over the optimum, (L - L*) / (1 - L*); the fraction of the optimal gain on each
    direction; and the loss with queries shifted toward rare directions."""
    with open(path) as f:
        data = json.load(f)
    lam, n = np.array(data["lam"]), data["n"]
    rho_star = lam * n / ((n + 1) * lam + lam.sum())
    L_star = 1 - np.sum(rho_star * lam) / len(lam)
    out = []
    for r in data["runs"]:
        loss = np.array(r["loss"])
        out.append(dict(arm=r["arm"], lr=r["lr"], seed=r["seed"], excess=(loss - L_star) / (1 - L_star),
                        gain=np.array(r["gain"]) / rho_star, qloss=np.array(r["qloss"])))
    return lam, out


def at_excess(r, values, level):
    """`values` where the run's excess loss first reaches `level`, interpolated linearly."""
    exc, v = r["excess"], np.asarray(values)
    i = int(np.nonzero(exc <= level)[0][0])
    if i == 0:
        return v[0]
    w = (exc[i - 1] - level) / (exc[i - 1] - exc[i])
    return v[i - 1] + w * (v[i] - v[i - 1])


def pcr_ridge_ratio(r, lam, levels=np.logspace(np.log10(0.5), np.log10(0.1), 8)):
    """How much better a ridge curve lam / (lam + a) fits the gain profile than a PCR step 1[j <= m]: the ratio of
    their least-squares residuals, averaged over excess losses 0.5 to 0.1 (above 1: ridge-like)."""
    shapes = lam / (lam + np.logspace(-3, 2, 200)[:, None])
    res = np.array([[min(np.sum((g - (np.arange(len(lam)) < m)) ** 2) for m in range(len(lam) + 1)),
                     np.min(np.sum((g - shapes) ** 2, 1))]
                    for g in (np.clip(at_excess(r, r["gain"], lv), 0, 1.2) for lv in levels)])
    return res[:, 0].mean() / res[:, 1].mean()
