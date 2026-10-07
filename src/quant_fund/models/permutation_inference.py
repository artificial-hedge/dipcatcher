"""Permutation / randomization inference (SYNTHETIC).

Fisher's exact randomization test for the sharp null of no effect —
the assignment mechanism is the *only* distribution invoked, so the
test is exact for any statistic and any sample size — plus
Westfall–Young step-down max-T familywise correction across multiple
statistics, and a randomization confidence interval obtained by
inverting the test over a constant-effect grid (Pitman's method).

All estimators fail closed (ValueError) on degenerate input; p-values
are Monte-Carlo approximations at resolution 1/(B+1) and reported as
such.

Honesty: synthetic benches measure size and power on generated A/B
data — never market evidence. Under the sharp null the size is
exactly nominal; the bench asserts control honestly (permutation p is
uniform under the null up to MC resolution).

References:
- Fisher (1935). *The Design of Experiments*. Oliver & Boyd.
- Pitman (1937). Significance tests which may be applied to samples
  from any populations. *JRSS Suppl.* 4.
- Westfall, Young (1993). *Resampling-Based Multiple Testing*. Wiley.
- Romano, Wolf (2005). Exact and approximate stepdown methods for
  multiple hypothesis testing. *JASA* 100.
- Imbens, Rubin (2015). *Causal Inference for Statistics, Social, and
  Biomedical Sciences* (Ch. 5, randomization inference).

Composition: pure numpy — no new dependencies; deterministic
``np.random.default_rng`` permutation draws.
"""

from __future__ import annotations

import math
from collections.abc import Callable

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


def _as_1d(x: FloatArray, name: str, n_min: int) -> FloatArray:
    a = np.asarray(x, dtype=np.float64).ravel()
    a = a[np.isfinite(a)]
    if a.size < n_min:
        raise ValueError(f"{name}: need >= {n_min} finite obs, got {a.size}")
    return a


def _diff_mean(y: FloatArray, tr: NDArray[np.bool_]) -> float:
    return float(y[tr].mean() - y[~tr].mean())


def fisher_permutation_p(
    y: FloatArray,
    treated: FloatArray,
    *,
    n_perm: int = 2000,
    seed: int = 0,
    statistic: Callable[[FloatArray, NDArray[np.bool_]], float] | None = None,
) -> dict[str, float]:
    """Two-sided randomization p-value for H0: treatment has no effect.

    ``treated`` is a boolean/0-1 assignment vector of the same length as
    ``y``; the observed statistic is compared against its permutation
    distribution under label reshuffling.
    """
    y = _as_1d(y, "y", 8)
    tr = np.asarray(treated).astype(bool).ravel()
    if tr.size != y.size or tr.sum() < 2 or tr.sum() > y.size - 2:
        raise ValueError("treated: need >= 2 treated and >= 2 control")
    if statistic is None:
        statistic = _diff_mean
    obs = float(statistic(y, tr))
    rng = np.random.default_rng(seed)
    n_tr = int(tr.sum())
    sims = np.empty(n_perm)
    idx = np.arange(y.size)
    for b in range(n_perm):
        perm = rng.permutation(idx)
        sims[b] = statistic(y, perm < n_tr)
    p = float((1.0 + np.sum(np.abs(sims) >= abs(obs) - 1e-12)) / (n_perm + 1.0))
    return {
        "stat_obs": obs,
        "p_two_sided": p,
        "perm_mean": float(sims.mean()),
        "perm_sd": float(sims.std(ddof=1)),
        "n_perm": float(n_perm),
        "mc_resolution": 1.0 / (n_perm + 1.0),
    }


def max_t_stepdown(
    stats_obs: FloatArray,
    stats_perm: FloatArray,
) -> dict[str, float | FloatArray]:
    """Westfall–Young step-down adjusted p-values from permutation draws.

    ``stats_obs`` is (m,) observed statistics; ``stats_perm`` is (B, m)
    permutation statistics under the joint null. Returns adjusted p-values
    enforcing strong FWER control under arbitrary dependence.
    """
    s = np.asarray(stats_obs, dtype=np.float64).ravel()
    sp = np.asarray(stats_perm, dtype=np.float64)
    if sp.ndim != 2 or sp.shape[1] != s.size or not np.all(np.isfinite(sp)):
        raise ValueError("stats_perm must be a finite (B, m) matrix")
    m = s.size
    if m < 1:
        raise ValueError("empty statistic vector")
    b = sp.shape[0]
    order = np.argsort(-np.abs(s))  # descending |t|
    adj = np.empty(m)
    running = 0.0
    for rank, j in enumerate(order):
        tail = sp[:, order[rank:]]
        max_t = np.max(np.abs(tail), axis=1)
        p_j = float((1.0 + np.sum(max_t >= abs(s[j]) - 1e-12)) / (b + 1.0))
        running = max(running, p_j)
        adj[j] = running
    return {
        "adjusted_p": adj,
        "min_adjusted": float(adj.min()),
        "n_stats": float(m),
    }


def randomization_ci(
    y: FloatArray,
    treated: FloatArray,
    *,
    grid_lo: float | None = None,
    grid_hi: float | None = None,
    n_grid: int = 81,
    n_perm: int = 500,
    alpha: float = 0.05,
    seed: int = 0,
) -> dict[str, float | FloatArray]:
    """Constant-effect confidence interval by test inversion.

    Under the hypothesis ``y_i(1) = y_i(0) + c``, shifting treated
    outcomes by ``-c`` makes the data sharp-null consistent; the set of
    non-rejected ``c`` is the randomization CI (Pitman inversion).
    """
    y = _as_1d(y, "y", 8)
    tr = np.asarray(treated).astype(bool).ravel()
    if tr.size != y.size or tr.sum() < 2 or tr.sum() > y.size - 2:
        raise ValueError("treated: need >= 2 treated and >= 2 control")
    if grid_lo is None or grid_hi is None:
        obs = _diff_mean(y, tr)
        span = max(4.0 * abs(obs), 2.0 * float(y.std()), 2.0)
        grid_lo, grid_hi = obs - span, obs + span
    grid = np.linspace(float(grid_lo), float(grid_hi), n_grid)
    inside = np.zeros(n_grid, dtype=bool)
    for i, c in enumerate(grid):
        yc = y.copy()
        yc[tr] -= c
        p = fisher_permutation_p(yc, tr.astype(np.float64), n_perm=n_perm, seed=seed + i)
        inside[i] = p["p_two_sided"] >= alpha
    idx = np.flatnonzero(inside)
    if idx.size == 0:
        lo, hi = math.nan, math.nan
    else:
        lo, hi = float(grid[idx[0]]), float(grid[idx[-1]])
    return {
        "ci_lo": lo,
        "ci_hi": hi,
        "ci_width": (hi - lo) if math.isfinite(hi - lo) else math.nan,
        "grid": grid,
        "inside": inside.astype(np.float64),
    }


def synth_ab(
    n: int = 400,
    frac_treated: float = 0.5,
    effect: float = 0.5,
    seed: int = 0,
) -> dict[str, FloatArray]:
    """A/B panel: treated arm shifted by ``effect`` plus iid noise."""
    rng = np.random.default_rng(seed)
    n_tr = int(round(n * frac_treated))
    tr = np.zeros(n, dtype=bool)
    tr[rng.choice(n, n_tr, replace=False)] = True
    y = rng.normal(0.0, 1.0, n) + tr * effect
    return {
        "y": y,
        "treated": tr.astype(np.float64),
        "effect": np.full(1, effect, dtype=np.float64),
    }


def bench_permutation_inference(seed: int = 20261231 + 188) -> dict[str, float]:
    """Permutation-inference self-check: exact size under the sharp null,
    power under a real shift, max-T correction, CI coverage.
    All ``synthetic_*``."""
    d = synth_ab(seed=seed, effect=0.6)
    y = np.asarray(d["y"])
    tr = np.asarray(d["treated"])

    p_eff = fisher_permutation_p(y, tr, n_perm=999, seed=seed + 1)
    d0 = synth_ab(seed=seed + 2, effect=0.0)
    p_null = fisher_permutation_p(
        np.asarray(d0["y"]), np.asarray(d0["treated"]), n_perm=999, seed=seed + 3
    )
    # size check across replicates: null p-values should be ~uniform
    rng_stats = []
    for j in range(25):
        dj = synth_ab(seed=seed + 10 + j, effect=0.0)
        pj = fisher_permutation_p(
            np.asarray(dj["y"]), np.asarray(dj["treated"]), n_perm=200, seed=seed + 50 + j
        )["p_two_sided"]
        rng_stats.append(pj)
    null_reject_rate = float(np.mean(np.array(rng_stats) < 0.05))

    ci = randomization_ci(y, tr, n_grid=61, n_perm=200, seed=seed + 4)

    # max-T: 3 stats, one true signal
    rng = np.random.default_rng(seed + 5)
    obs = np.array([3.0, 0.4, -0.2])
    null_draws = rng.normal(0.0, 1.0, (500, 3)) * np.array([[1.0, 1.0, 1.0]])
    mt = max_t_stepdown(obs, null_draws)

    return {
        "synthetic_p_effect": float(p_eff["p_two_sided"]),
        "synthetic_p_null": float(p_null["p_two_sided"]),
        "synthetic_null_reject_rate": null_reject_rate,
        "synthetic_size_controlled": float(null_reject_rate <= 0.20),
        "synthetic_power": float(p_eff["p_two_sided"] < 0.01),
        "synthetic_ci_covers": float(
            math.isfinite(float(ci["ci_lo"])) and float(ci["ci_lo"]) <= 0.6 <= float(ci["ci_hi"])
        ),
        "synthetic_ci_width": float(ci["ci_width"]),
        "synthetic_max_t_signal": float(np.asarray(mt["adjusted_p"])[0] < 0.05),
        "synthetic_max_t_null_loose": float(
            np.asarray(mt["adjusted_p"])[1] > np.asarray(mt["adjusted_p"])[0]
        ),
        "synthetic_determinism": float(
            fisher_permutation_p(y, tr, n_perm=999, seed=seed + 1)["p_two_sided"]
            == p_eff["p_two_sided"]
        ),
    }
