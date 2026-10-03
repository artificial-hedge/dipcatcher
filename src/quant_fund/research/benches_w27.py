"""Benchmark batteries for SOTA canon wave 27 (see waves 11-26 for the pattern).

Wave 27 lands OPTIONAL research families, each pinned to the work shipped
in its lane module (see the module docstrings for full citations):

- ``durbin_koopman``: Durbin-Koopman state-space machinery — NaN-masked
  Kalman filter, RTS smoother, disturbance smoother, simulation smoother
  for posterior path draws, marginal coverage on synthetic SSM.
- ``dp_mixture``: Dirichlet-process Gaussian mixture regime discovery —
  Blei-Jordan CAVI on a truncated stick-breaking representation, ELBO
  ascent, effective-component count, adjusted-Rand recovery.
- ``expert_aggregation``: prediction with expert advice — Hedge/EWA,
  Herbster-Warmuth fixed-share tracking, sleeping specialists,
  regret-vs-bounds on switching synthetic panels.
- ``instrumental_quantile``: Chernozhukov-Hansen IVQR — grid-inverted
  instrument projection, Anderson-Rubin weak-IV-robust tests, first-stage F.
- ``implied_tree``: Derman-Kani implied binomial tree — alternating-center
  smile calibration, Arrow-Debreu densities, martingale checks, absorbing
  knockout pricing.
- ``ensemble_kalman_inversion``: Ensemble Kalman Inversion — stochastic EKI
  with perturbed observations, Tikhonov-regularized variant, linear
  recovery + SV moment matching.

Honesty contract: all blobs are SYNTHETIC correctness/telemetry checks, never
market evidence. No Sharpe/Sortino/Calmar/PnL/NAV tokens appear in emitted
keys; every bench returns ``{}`` (fails soft) if any emitted value is
non-finite — the ``_finite_blob`` gate, waves 15-26 precedent.
"""

from __future__ import annotations

import math

_SEED = 20261231  # wave-27 stamp seed

# --- per-family seeds -------------------------------------------------------
_DK_SEED = _SEED + 141
_DPMM_SEED = _SEED + 142
_EXPERT_SEED = _SEED + 143
_IVQR_SEED = _SEED + 144
_TREE_SEED = _SEED + 145
_EKI_SEED = _SEED + 146


def _finite_blob(mapped: dict[str, float]) -> dict[str, float]:
    """``{}`` unless every emitted value is finite (ruff-bench contract)."""
    if all(math.isfinite(v) for v in mapped.values()):
        return mapped
    return {}


_FORBIDDEN_TOKENS = frozenset({"sharpe", "sortino", "calmar", "pnl", "nav"})


def _isinstance_floats(
    raw: dict[str, float] | dict[str, object],
) -> dict[str, float]:
    """Float-coerce a lane blob, dropping str stamps, runtime telemetry,
    and any key carrying a forbidden headline metric token."""
    return {
        k: float(v)
        for k, v in raw.items()
        if isinstance(v, (int, float)) and _FORBIDDEN_TOKENS.isdisjoint(k.lower().split("_"))
    }


def bench_durbin_koopman() -> dict[str, float]:
    """Durbin-Koopman state-space bench (wave 27).

    Smoothed-state correlation vs truth, disturbance-smoother recovery,
    simulation-smoother marginal coverage vs nominal, loglik sanity.
    Soft ``{}`` while absent.
    """
    try:
        from quant_fund.models.durbin_koopman import (
            bench_durbin_koopman as _core,
        )
    except ImportError:
        return {}
    try:
        return _finite_blob(_isinstance_floats(_core(seed=_DK_SEED)))
    except (ValueError, RuntimeError, FloatingPointError, KeyError, TypeError):
        return {}


def bench_dp_mixture() -> dict[str, float]:
    """DP-mixture regime discovery bench (wave 27).

    Adjusted Rand vs planted regimes, effective-component recovery, ELBO
    monotonicity, assignment sharpness, predictive density. Soft ``{}``
    while absent.
    """
    try:
        from quant_fund.models.dp_mixture import bench_dp_mixture as _core
    except ImportError:
        return {}
    try:
        return _finite_blob(_isinstance_floats(_core(seed=_DPMM_SEED)))
    except (ValueError, RuntimeError, FloatingPointError, KeyError, TypeError):
        return {}


def bench_expert_aggregation() -> dict[str, float]:
    """Expert-advice aggregation bench (wave 27).

    Hedge regret ratio vs bound, fixed-share beats static under switches,
    switch-detect lag, specialist edge, determinism. Soft ``{}`` while
    absent.
    """
    try:
        from quant_fund.models.expert_aggregation import (
            bench_expert_aggregation as _core,
        )
    except ImportError:
        return {}
    try:
        return _finite_blob(_isinstance_floats(_core(seed=_EXPERT_SEED)))
    except (ValueError, RuntimeError, FloatingPointError, KeyError, TypeError):
        return {}


def bench_instrumental_quantile() -> dict[str, float]:
    """IVQR bench (wave 27).

    IVQR vs naive-QR bias under endogeneity, Anderson-Rubin size/power,
    first-stage F, CI coverage. Soft ``{}`` while absent.
    """
    try:
        from quant_fund.metrics.instrumental_quantile import (
            bench_instrumental_quantile as _core,
        )
    except ImportError:
        return {}
    try:
        return _finite_blob(_isinstance_floats(_core(seed=_IVQR_SEED)))
    except (ValueError, RuntimeError, FloatingPointError, KeyError, TypeError):
        return {}


def bench_implied_tree() -> dict[str, float]:
    """Implied-tree smile calibration bench (wave 27).

    Smile repricing error, Arrow-Debreu mass, martingale error, arbitrage
    violation count, knockout bounds. Soft ``{}`` while absent.
    """
    try:
        from quant_fund.models.implied_tree import (
            bench_implied_tree as _core,
        )
    except ImportError:
        return {}
    try:
        return _finite_blob(_isinstance_floats(_core(seed=_TREE_SEED)))
    except (ValueError, RuntimeError, FloatingPointError, KeyError, TypeError):
        return {}


def bench_ensemble_kalman_inversion() -> dict[str, float]:
    """EKI bench (wave 27).

    Parameter recovery error on a linear inverse problem, misfit
    reduction, spread collapse, SV moment matching, determinism.
    Soft ``{}`` while absent.
    """
    try:
        from quant_fund.models.ensemble_kalman_inversion import (
            bench_ensemble_kalman_inversion as _core,
        )
    except ImportError:
        return {}
    try:
        return _finite_blob(_isinstance_floats(_core(seed=_EKI_SEED)))
    except (ValueError, RuntimeError, FloatingPointError, KeyError, TypeError):
        return {}
