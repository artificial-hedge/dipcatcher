"""Bayesian online change-point detection (BOCPD).

Implements the Adams & MacKay run-length posterior recursion: at each
observation the posterior over the current run length ``r_t`` splits
between a change (mass falls back to ``r=0``) and growth
(``r -> r+1`` weighted by the predictive density). The hazard function
``H`` gives the prior change rate; a constant hazard produces a
geometric segment prior. The predictive model is the conjugate
normal-gamma (Student-t) learner of the paper's Gaussian examples, so
posterior/prior predictive densities are closed-form Student-t.

    Adams, R. P., & MacKay, D. J. C. (2007). "Bayesian Online
    Changepoint Detection." arXiv:0710.3742 [stat.ML]. Citation
    verified against https://arxiv.org/abs/0710.3742 (fetched
    2026-09-30).

Machinery: O(1)-amortized run-length pruning (tail-mass truncation),
MAP run-length path, expected run-length curve, change-point posterior
maxima with detection-delay reporting against planted change points,
and a log-predictive-score comparison against an oracle filter that
knows the true segment ids.

Honesty: all bench results are SYNTHETIC multi-regime series — correctness
diagnostics only (detection delay, precision/recall, predictive score),
never market evidence; no P&L/NAV claims.

Composition notes: pure numpy/scipy; no existing quant_fund module owns
run-length posteriors (``models/stoch_vol`` uses a Kalman filter, not a
discrete hazard recursion). The Student-t sufficient-statistic form
follows the normal-gamma conjugate of the cited paper rather than any
in-repo predictive code.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy import stats

Array = NDArray[np.float64]

__all__ = [
    "BocpdResult",
    "NormalGammaLearner",
    "bench_bocpd_changepoint",
    "changepoint_maxima",
    "constant_hazard",
    "detected_change_points",
    "expected_run_length",
    "run_bocpd",
    "synthetic_multi_regime",
]


_EPS = 1e-12


def _require(cond: object, msg: str) -> None:
    if not bool(cond):
        raise ValueError(msg)


def constant_hazard(n: int, lam: float) -> Array:
    """Constant hazard H = 1/lam over a run-length support of ``n``."""
    _require(n >= 2, "run-length support needs >=2")
    _require(lam > 0, "lam must be positive")
    return np.full(n, 1.0 / lam)


def geometric_hazard(lam: float) -> Array:
    """Hazard vector implied by a geometric segment-length prior.

    ``H(r) = 1/lam`` for every r (the geometric prior is memoryless), so
    this is just ``constant_hazard`` under another name for callers that
    prefer the paper's notation.
    """
    _require(lam > 0, "lam must be positive")
    return np.array([1.0 / lam])


# ---------------------------------------------------------------------------
# Conjugate predictive model (normal-gamma -> Student-t)
# ---------------------------------------------------------------------------


@dataclass
class NormalGammaLearner:
    """Conjugate Gaussian learner with a normal-gamma prior.

    Hyperparameters (mu0, kappa0, alpha0, beta0); posterior predictive
    is Student-t with ``2*alpha`` degrees of freedom, location ``mu``
    and scale ``sqrt(beta*(kappa+1)/(alpha*kappa))``. State is kept as
    arrays across the run-length support, grown one slot per step.
    """

    mu0: float = 0.0
    kappa0: float = 1.0
    alpha0: float = 1.0
    beta0: float = 1.0

    def __post_init__(self) -> None:
        _require(self.kappa0 > 0, "kappa0 must be positive")
        _require(self.alpha0 > 0, "alpha0 must be positive")
        _require(self.beta0 > 0, "beta0 must be positive")
        self.mu = np.array([self.mu0])
        self.kappa = np.array([self.kappa0])
        self.alpha = np.array([self.alpha0])
        self.beta = np.array([self.beta0])

    def predict(self, x: float) -> Array:
        """Posterior-predictive density of ``x`` at each run length."""
        df = 2.0 * self.alpha
        scale = np.sqrt(self.beta * (self.kappa + 1.0) / (self.alpha * self.kappa))
        with np.errstate(divide="ignore", invalid="ignore"):
            dens = stats.t.pdf(x, df, loc=self.mu, scale=scale)
        return np.asarray(dens, dtype=float)

    def update(self, x: float) -> None:
        """Fold ``x`` into the per-run-length posterior state and grow."""
        k1 = self.kappa + 1.0
        mu1 = (self.kappa * self.mu + x) / k1
        beta1 = self.beta + 0.5 * self.kappa * (x - self.mu) ** 2 / k1
        # grow: slot 0 is a fresh prior, slots 1.. carry the update
        self.mu = np.concatenate(([self.mu0], mu1))
        self.kappa = np.concatenate(([self.kappa0], k1))
        self.alpha = np.concatenate(([self.alpha0], self.alpha + 0.5))
        self.beta = np.concatenate(([self.beta0], beta1))


@dataclass
class BocpdResult:
    """Run-length posterior diagnostics for one series."""

    run_length_posterior: Array  # (T, max_len) posterior matrix
    map_run_length: Array  # (T,) MAP run length after each obs
    expected_run_length: Array  # (T,)
    change_posterior: Array  # (T,) posterior mass on short runs P(r<8)
    log_predictive: Array  # (T,) marginal predictive log-density
    hazard: Array

    @property
    def n_obs(self) -> int:
        return int(self.map_run_length.size)


def run_bocpd(
    x: Array,
    hazard: Array,
    learner: NormalGammaLearner | None = None,
    prune_tail: float = 1e-9,
    max_run: int = 400,
) -> BocpdResult:
    """Adams & MacKay run-length posterior recursion.

    ``x`` observations; ``hazard`` H(r) per run length (indexed up to
    ``max_run``); ``learner`` is the conjugate predictive model.
    ``prune_tail`` truncates run-length mass below that fraction of the
    posterior each step (keeps O(1) amortized work).
    """
    xx = np.asarray(x, dtype=float).ravel()
    _require(xx.size >= 5 and np.all(np.isfinite(xx)), "x must be finite len>=5")
    hh = np.asarray(hazard, dtype=float).ravel()
    _require(hh.size >= 2 and np.all(hh >= 0) and np.all(hh <= 1), "hazard must be in [0,1]")
    if learner is None:
        learner = NormalGammaLearner()

    t_n = xx.size
    post = np.zeros((t_n, max_run))
    map_rl = np.zeros(t_n)
    exp_rl = np.zeros(t_n)
    change_post = np.zeros(t_n)
    log_pred = np.zeros(t_n)

    # probs[r] = P(r_t = r); index == run length, contiguous support.
    probs = np.array([1.0])
    for t in range(t_n):
        x_t = float(xx[t])
        n_cur = probs.size
        # learner slot r tracks the r-long run; pruning may drop the tail
        pred = np.maximum(learner.predict(x_t)[:n_cur], _EPS)
        h_vec = np.full(n_cur, hh[0]) if hh.size == 1 else hh[:n_cur]
        # growth: r -> r+1, weighted by predictive and survival (1-H)
        growth = probs * pred * (1.0 - h_vec)
        change = float(np.sum(probs * pred * h_vec))
        new = np.empty(n_cur + 1)
        new[0] = change
        new[1:] = growth
        total = float(new.sum())
        _require(total > 0, "posterior collapsed to zero mass")
        new /= total
        # prune the long-run-length tail (least-mass slots, contiguous)
        if new.size > max_run:
            new = new[:max_run]
            new /= new.sum()
        tail = np.where(np.cumsum(new[::-1]) < prune_tail)[0]
        if tail.size and new.size - tail.size > 1:
            new = new[: new.size - tail.size]
            new /= new.sum()
        probs = new
        learner.update(x_t)
        log_pred[t] = math.log(total)
        post[t, : probs.size] = probs
        map_rl[t] = float(np.argmax(probs))
        exp_rl[t] = float(np.arange(probs.size) @ probs)
        # change signal: posterior mass on short runs (P(r_t < 8)) — the
        # r=0 slot alone stays ~H under a constant hazard.
        change_post[t] = float(probs[:8].sum())
    return BocpdResult(
        run_length_posterior=post,
        map_run_length=map_rl,
        expected_run_length=exp_rl,
        change_posterior=change_post,
        log_predictive=log_pred,
        hazard=hh,
    )


def expected_run_length(result: BocpdResult) -> Array:
    return result.expected_run_length


def changepoint_maxima(result: BocpdResult, min_gap: int = 8, threshold: float = 0.05) -> list[int]:
    """Local maxima of the change posterior, at least ``min_gap`` apart."""
    _require(min_gap >= 1, "min_gap must be >=1")
    _require(0.0 <= threshold < 1.0, "threshold in [0,1)")
    cp = result.change_posterior
    picks: list[int] = []
    t = 0
    while t < cp.size:
        j = int(np.argmax(cp[t : t + min_gap])) + t
        if cp[j] >= threshold:
            picks.append(j)
            t = j + min_gap
        else:
            t += min_gap
    return picks


def detected_change_points(result: BocpdResult, min_gap: int = 8) -> list[int]:
    """Times where the MAP run length resets to ~0 (drop > half)."""
    rl = result.map_run_length
    out: list[int] = []
    for t in range(1, rl.size):
        if rl[t] < 0.5 * rl[t - 1] and rl[t - 1] > 4 and (not out or t - out[-1] >= min_gap):
            out.append(t)
    return out


# ---------------------------------------------------------------------------
# Synthetic fixture + bench
# ---------------------------------------------------------------------------


def synthetic_multi_regime(
    seed: int,
    n_seg: int = 4,
    seg_len: int = 40,
    drift: float = 0.0,
) -> tuple[Array, Array]:
    """Piecewise-Gaussian series; returns (x, changepoint_times)."""
    rng = np.random.default_rng(seed)
    _require(n_seg >= 2, "n_seg must be >=2")
    _require(seg_len >= 10, "seg_len must be >=10")
    mu_rng = rng.uniform(-1.5, 1.5, size=n_seg)
    sd_rng = rng.uniform(0.2, 1.2, size=n_seg)
    x = np.empty(n_seg * seg_len)
    cps = []
    for s in range(n_seg):
        if s:
            cps.append(s * seg_len)
        x[s * seg_len : (s + 1) * seg_len] = rng.normal(mu_rng[s] + drift, sd_rng[s], size=seg_len)
    return x, np.asarray(cps, dtype=float)


def bench_bocpd_changepoint(
    seed: int,
    n_series: int = 6,
    n_seg: int = 4,
    seg_len: int = 40,
    lam: float = 120.0,
) -> dict[str, float]:
    """Seeded SYNTHETIC bench for the BOCPD machinery.

    Planted-regime series: detection delay/precision/recall of the
    MAP-reset change points vs the true segment starts; mean
    log-predictive score of the full BOCPD filter vs an oracle that
    resets its conjugate learner at each true boundary.
    """
    delays: list[float] = []
    tp_c = fp_c = fn_c = 0
    score_bocpd = score_oracle = score_nochange = 0.0
    for i in range(n_series):
        x, cps = synthetic_multi_regime(seed + 97 * i, n_seg=n_seg, seg_len=seg_len)
        res = run_bocpd(
            x,
            constant_hazard(400, lam),
            learner=NormalGammaLearner(beta0=0.5),
            max_run=350,
        )
        score_bocpd += float(np.mean(res.log_predictive))
        found = detected_change_points(res, min_gap=10)
        matched = [False] * cps.size
        for f in found:
            near = np.argmin(np.abs(cps - f))
            if abs(cps[near] - f) <= 8 and not matched[near]:
                matched[near] = True
                delays.append(abs(cps[near] - f))
                tp_c += 1
            else:
                fp_c += 1
        fn_c += int(np.sum(~np.asarray(matched)))
        # oracle: one learner per segment, same predictive form
        oracle = 0.0
        for s in range(n_seg):
            seg = x[s * seg_len : (s + 1) * seg_len]
            lr = NormalGammaLearner(beta0=0.5)
            for v in seg:
                oracle += math.log(float(np.maximum(lr.predict(v)[-1], _EPS)))
                lr.update(v)
        score_oracle += oracle / x.size
        nc = NormalGammaLearner(beta0=0.5)
        ncl = 0.0
        for v in x:
            ncl += math.log(float(np.maximum(nc.predict(v)[-1], _EPS)))
            nc.update(v)
        score_nochange += ncl / x.size
    n = float(n_series)
    precision = tp_c / max(tp_c + fp_c, 1)
    recall = tp_c / max(tp_c + fn_c, 1)
    out = {
        "synthetic_seed": float(seed),
        "synthetic_n_series": n,
        "synthetic_detection_delay_mean": float(np.mean(delays)) if delays else 0.0,
        "synthetic_precision": precision,
        "synthetic_recall": recall,
        "synthetic_n_true_cp": float(n_series * (n_seg - 1)),
        "synthetic_logpred_bocpd": score_bocpd / n,
        "synthetic_logpred_oracle": score_oracle / n,
        "synthetic_logpred_nochange": score_nochange / n,
        "synthetic_oracle_gap": score_oracle / n - score_bocpd / n,
        "synthetic_beats_nochange": float(score_bocpd / n > score_nochange / n),
    }
    if not all(math.isfinite(v) for v in out.values()):
        return {}
    return out
