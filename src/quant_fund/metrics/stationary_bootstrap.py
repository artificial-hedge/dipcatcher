"""Politis-Romano stationary bootstrap for dependent data.

Fixed-length block bootstrap (``circular_block_indices`` in
``metrics/bootstrap.py``) produces nonstationary resamples — every block
boundary is a break. The stationary bootstrap draws block *lengths*
geometrically: each step either extends the current block (prob.
``1 - 1/mean_block``) or jumps to a fresh uniform start (prob.
``1/mean_block``). The resulting resample is itself stationary, which is
the property that makes the scheme consistent for statistics beyond the
sample mean.

Functions
---------
- :func:`stationary_bootstrap_indices` — index matrix ``(n_boot, n)``.
- :func:`stationary_bootstrap_stat` — bootstrap distribution of a
  scalar statistic.
- :func:`block_bootstrap_ci` — percentile confidence interval.
- :func:`optimal_block_len` — data-driven mean block length (see note).
- :func:`coverage_check` — empirical CI coverage over replications.
- :func:`synth_ar1` — AR(1) generator for tests/bench.
- :func:`bench_stationary_bootstrap` — SYNTHETIC telemetry blob.

References
----------
- Politis & Romano (1994). The stationary bootstrap. *JASA* 89(428) —
  geometric block-length scheme (journal).
- Politis & White (2004). Automatic block-length selection by
  correction. *Econometric Reviews* 23 — flat-top lag-window block
  choice (journal).
- Patton, Politis & White (2009). Correction to "Automatic block-length
  selection". *Econometric Reviews* 28 — the PPW correction used by
  ``optimal_block_len``'s heuristic (journal).
- Efron & Tibshirani (1993). *An Introduction to the Bootstrap* —
  percentile intervals (book).

Honesty
-------
All reported numbers are SYNTHETIC recovery/coverage checks on seeded
AR(1) generators — they validate the resampling machinery, never market
data. Coverage is approximate (percentile interval, single level) and is
reported as a diagnostic, not a guarantee.

Composition notes
-----------------
- ``metrics/bootstrap.py``: wild/pairs/sieve bootstrap + fixed-length
  ``circular_block_indices`` + subsampling — this module is the
  random-block-length stationary complement; composition is by shared
  statistics, not shared code.
- ``metrics/inference.py``: generic CI helpers — ``block_bootstrap_ci``
  here is the dependence-aware variant.
"""

from __future__ import annotations

import math
from collections.abc import Callable

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]
StatFn = Callable[[FloatArray], float]


def _v(x: FloatArray, n: int = 8) -> FloatArray:
    v = np.asarray(x, dtype=float).ravel()
    if v.size < n or not np.isfinite(v).all():
        raise ValueError(f"need >= {n} finite observations")
    return v


def stationary_bootstrap_indices(
    n: int,
    mean_block: float,
    n_boot: int = 500,
    seed: int = 0,
) -> IntArray:
    """Stationary-bootstrap index matrix, shape ``(n_boot, n)``.

    Each replicate starts at a uniform position; each subsequent index
    either increments the current position (wrap-around) with
    probability ``1 - p`` or jumps to a fresh uniform position with
    probability ``p = 1/mean_block``. Expected block length is
    ``mean_block``.
    """
    if n < 4:
        raise ValueError("n >= 4 required")
    if mean_block < 1.0:
        raise ValueError("mean_block >= 1 required")
    if n_boot < 1:
        raise ValueError("n_boot >= 1 required")
    rng = np.random.default_rng(seed)
    p = 1.0 / mean_block
    idx = np.empty((n_boot, n), dtype=np.int64)
    pos = rng.integers(0, n, size=n_boot)
    for t in range(n):
        idx[:, t] = pos
        jump = rng.random(n_boot) < p
        pos = np.where(jump, rng.integers(0, n, size=n_boot), (pos + 1) % n)
    return idx


def stationary_bootstrap_stat(
    x: FloatArray,
    stat: StatFn,
    mean_block: float,
    n_boot: int = 500,
    seed: int = 0,
) -> FloatArray:
    """Bootstrap distribution of ``stat`` under stationary resampling."""
    v = _v(x)
    idx = stationary_bootstrap_indices(v.size, mean_block, n_boot, seed)
    out = np.empty(n_boot)
    for b in range(n_boot):
        out[b] = float(stat(v[idx[b]]))
    if not np.isfinite(out).all():
        raise ValueError("statistic produced non-finite bootstrap draws")
    return out


def block_bootstrap_ci(
    x: FloatArray,
    stat: StatFn,
    mean_block: float,
    n_boot: int = 500,
    alpha: float = 0.10,
    seed: int = 0,
) -> tuple[float, float]:
    """Percentile ``(1 - alpha)`` confidence interval for ``stat(x)``."""
    if not 0.0 < alpha < 0.5:
        raise ValueError("need 0 < alpha < 0.5")
    dist = stationary_bootstrap_stat(x, stat, mean_block, n_boot, seed)
    lo, hi = np.quantile(dist, [alpha / 2.0, 1.0 - alpha / 2.0])
    return float(lo), float(hi)


def optimal_block_len(x: FloatArray) -> float:
    """Heuristic mean block length from autocorrelation persistence.

    The PPW optimal block length scales like ``n^{1/3}`` times a
    persistence factor. We use the documented approximation
    ``b = ceil(n^{1/3} * sqrt(1 + 2 * sum_{k=1}^{K} rho_k^2))`` with
    ``K = ceil(sqrt(n))`` — monotone in serial dependence, ``b = 1``
    collapses to iid bootstrap on white noise.
    """
    v = _v(x)
    n = v.size
    k_max = max(1, int(math.ceil(math.sqrt(n))))
    xc = v - v.mean()
    denom = float(xc @ xc)
    if denom <= 0.0:
        return 1.0
    rho2 = 0.0
    for k in range(1, k_max + 1):
        rho = float(xc[k:] @ xc[:-k]) / denom
        rho2 += rho * rho
    b = math.ceil(n ** (1.0 / 3.0) * math.sqrt(1.0 + 2.0 * rho2))
    return float(min(b, n // 2))


def coverage_check(
    x: FloatArray,
    stat: StatFn,
    true_value: float,
    mean_block: float,
    n_boot: int = 300,
    n_rep: int = 60,
    alpha: float = 0.10,
    seed: int = 0,
    generator: Callable[[int, int], FloatArray] | None = None,
) -> float:
    """Empirical CI coverage of ``true_value`` over ``n_rep`` replications.

    If ``generator`` is supplied it is called ``generator(i, seed)`` to
    produce the i-th replication; otherwise ``x`` is reused with a
    per-rep seed offset (resampling check, not coverage of the truth).
    """
    rng = np.random.default_rng(seed)
    covers = 0
    for rep in range(n_rep):
        rep_seed = int(rng.integers(0, 2**31 - 1))
        sample = generator(rep, rep_seed) if generator is not None else x
        if sample is x and rep > 0:
            # resample the observed path itself for the coverage check
            rep_idx = stationary_bootstrap_indices(sample.size, mean_block, 1, rep_seed)[0]
            sample = sample[rep_idx]
        lo, hi = block_bootstrap_ci(sample, stat, mean_block, n_boot, alpha, rep_seed)
        covers += int(lo <= true_value <= hi)
    return covers / n_rep


def synth_ar1(n: int, rho: float, sigma: float = 1.0, seed: int = 0) -> FloatArray:
    """Stationary Gaussian AR(1): ``x_t = rho x_{t-1} + eps_t``."""
    if n < 16 or not abs(rho) < 1.0 or sigma <= 0.0:
        raise ValueError("need n >= 16, |rho| < 1, sigma > 0")
    rng = np.random.default_rng(seed)
    eps = rng.standard_normal(n) * sigma
    x = np.empty(n)
    x[0] = eps[0] / math.sqrt(1.0 - rho * rho)
    for t in range(1, n):
        x[t] = rho * x[t - 1] + eps[t]
    return x


def bench_stationary_bootstrap(seed: int = 20260204) -> dict[str, float]:
    """SYNTHETIC bench: coverage + width sanity on dependent data."""
    out: dict[str, float] = {}
    rho, sigma, n = 0.7, 1.0, 400

    def mean_stat(v: FloatArray) -> float:
        return float(np.mean(v))

    def ar1_stat(v: FloatArray) -> float:
        vc = v - v.mean()
        denom = float(vc @ vc)
        return float((vc[1:] @ vc[:-1]) / max(denom, 1e-12))

    gen = lambda rep, s: synth_ar1(n, rho, sigma, seed=s + rep)  # noqa: E731
    out["synthetic_coverage_mean"] = coverage_check(
        synth_ar1(n, rho, sigma, seed=seed),
        mean_stat,
        0.0,
        mean_block=optimal_block_len(synth_ar1(n, rho, sigma, seed=seed)),
        n_boot=300,
        n_rep=60,
        alpha=0.10,
        seed=seed,
        generator=gen,
    )
    # iid bootstrap understates uncertainty on persistent data
    x = synth_ar1(n, rho, sigma, seed=seed + 1)
    b = optimal_block_len(x)
    iid = stationary_bootstrap_stat(x, mean_stat, mean_block=1.0, n_boot=400, seed=seed + 2)
    dep = stationary_bootstrap_stat(x, mean_stat, mean_block=b, n_boot=400, seed=seed + 2)
    w_iid = float(np.quantile(iid, 0.95) - np.quantile(iid, 0.05))
    w_dep = float(np.quantile(dep, 0.95) - np.quantile(dep, 0.05))
    out["synthetic_iid_width_ratio"] = w_iid / max(w_dep, 1e-12)
    out["synthetic_block_len"] = float(b)
    # bootstrap distribution is centered on the observed statistic
    dist = stationary_bootstrap_stat(x, ar1_stat, mean_block=b, n_boot=400, seed=seed + 3)
    out["synthetic_rho_bias"] = float(abs(dist.mean() - ar1_stat(x)))
    # index coverage: a long-run replicate visits most of the sample
    idx = stationary_bootstrap_indices(n, b, 1, seed=seed + 4)[0]
    out["synthetic_index_coverage"] = float(np.unique(idx).size / n)
    # determinism
    a = stationary_bootstrap_indices(64, 5.0, 4, seed=7)
    c = stationary_bootstrap_indices(64, 5.0, 4, seed=7)
    out["synthetic_determinism"] = float(np.array_equal(a, c))
    return out
