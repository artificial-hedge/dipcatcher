"""Benchmark batteries for SOTA canon wave 24 (see waves 11-23 for the pattern).

Wave 24 lands OPTIONAL research families, each pinned to the paper shipped
in its lane module (see the module docstrings for full citations; citation
ids verified against arXiv before implementation):

- ``pmcmc_sv``: particle-MCMC stochastic-volatility estimation —
  marginal-likelihood particle filter, PMCMC sampler, state-posterior
  recovery on synthetic SV data.
- ``multifractal_vol``: multifractal volatility diagnostics — partition
  functions, scaling exponents, singularity spectra, cascade simulation.
- ``spci_conformal``: SPCI online quantile regression — IRLS pinball
  ridge walk-forward with AdaptiveConformal comparison (Xu-Xie SPCI;
  see module docstring).
- ``hawkes_em``: EM estimation of multivariate Hawkes branching —
  immigrant/offspring responsibility E-step, spectral-radius branching
  diagnostics, vs direct MLE (see module docstring).
- ``fernholz_spt``: stochastic portfolio theory — market entropy,
  functionally-generated portfolio drift decomposition, rank-process
  local times, diversity-vs-cap master equations (Fernholz SPT
  literature — see module docstring).
- ``breeden_litzenberger``: risk-neutral density extraction — BL second
  derivative on smoothed smiles, arbitrage checks, implied moments
  (see module docstring).

Honesty contract: all blobs are SYNTHETIC correctness/telemetry checks, never
market evidence. No Sharpe/Sortino/Calmar/PnL/NAV tokens appear in emitted
keys; every bench returns ``{}`` (fails soft) if any emitted value is
non-finite — the ``_finite_blob`` gate, waves 15-23 precedent.
"""

from __future__ import annotations

import math

_SEED = 20261228  # wave-24 stamp seed

# --- per-family seeds -------------------------------------------------------
_PMCCMC_SEED = _SEED + 111
_MULTI_SEED = _SEED + 112
_SPCI_SEED = _SEED + 113
_HAWKES_SEED = _SEED + 114
_FERNHOLZ_SEED = _SEED + 115
_BREEDEN_SEED = _SEED + 116


def _finite_blob(mapped: dict[str, float]) -> dict[str, float]:
    """``{}`` unless every emitted value is finite (ruff-bench contract)."""
    if all(math.isfinite(v) for v in mapped.values()):
        return mapped
    return {}


_FORBIDDEN_TOKENS = frozenset({"sharpe", "sortino", "calmar", "pnl", "nav"})


def _isinstance_floats(raw: dict[str, float] | dict[str, object]) -> dict[str, float]:
    """Float-coerce a lane blob, dropping str stamps, runtime telemetry,
    and any key carrying a forbidden headline metric token."""
    return {
        k: float(v)
        for k, v in raw.items()
        if isinstance(v, (int, float)) and _FORBIDDEN_TOKENS.isdisjoint(k.lower().split("_"))
    }


def bench_pmcmc_sv() -> dict[str, float]:
    """Particle-MCMC stochastic-volatility bench (wave 24).

    Marginal-likelihood scores, posterior mean-volatility recovery,
    sampler diagnostics. Soft ``{}`` while absent.
    """
    try:
        from quant_fund.models.pmcmc_sv import bench_pmcmc_sv as _core
    except ImportError:
        return {}
    try:
        return _finite_blob(_isinstance_floats(_core(seed=_PMCCMC_SEED)))
    except (ValueError, RuntimeError, FloatingPointError, KeyError, TypeError):
        return {}


def bench_multifractal_vol() -> dict[str, float]:
    """Multifractal volatility bench (wave 24).

    Partition-function scaling, singularity-spectrum recovery, cascade
    simulation diagnostics. Soft ``{}`` while absent.
    """
    try:
        from quant_fund.models.multifractal_vol import (
            bench_multifractal_vol as _core,
        )
    except ImportError:
        return {}
    try:
        return _finite_blob(_isinstance_floats(_core(seed=_MULTI_SEED)))
    except (ValueError, RuntimeError, FloatingPointError, KeyError, TypeError):
        return {}


def bench_spci_conformal() -> dict[str, float]:
    """SPCI online quantile-regression bench (wave 24).

    Coverage/width on drifted synthetic series, ACI comparator scores,
    pinball-ridge residual diagnostics. Soft ``{}`` while absent.
    """
    try:
        from quant_fund.models.spci_conformal import bench_spci_conformal as _core
    except ImportError:
        return {}
    try:
        return _finite_blob(_isinstance_floats(_core(seed=_SPCI_SEED)))
    except (ValueError, RuntimeError, FloatingPointError, KeyError, TypeError):
        return {}


def bench_hawkes_em() -> dict[str, float]:
    """EM Hawkes branching bench (wave 24).

    EM-vs-MLE branching-matrix recovery, spectral-radius diagnostics,
    log-likelihood trace monotone checks. Soft ``{}`` while absent.
    """
    try:
        from quant_fund.models.hawkes_em import bench_hawkes_em as _core
    except ImportError:
        return {}
    try:
        return _finite_blob(_isinstance_floats(_core(seed=_HAWKES_SEED)))
    except (ValueError, RuntimeError, FloatingPointError, KeyError, TypeError):
        return {}


def bench_fernholz_spt() -> dict[str, float]:
    """Stochastic-portfolio-theory bench (wave 24).

    FG-portfolio master-equation residuals, entropy/diversity recovery,
    local-time estimator errors, horizon diagnostics. Soft ``{}`` while
    absent.
    """
    try:
        from quant_fund.metrics.fernholz_spt import bench_fernholz_spt as _core
    except ImportError:
        return {}
    try:
        return _finite_blob(_isinstance_floats(_core(seed=_FERNHOLZ_SEED)))
    except (ValueError, RuntimeError, FloatingPointError, KeyError, TypeError):
        return {}


def bench_breeden_litzenberger() -> dict[str, float]:
    """Risk-neutral density bench (wave 24).

    L1 density recovery, implied-moment errors, mass-defect repair,
    noise robustness. Soft ``{}`` while absent.
    """
    try:
        from quant_fund.models.breeden_litzenberger import (
            bench_breeden_litzenberger as _core,
        )
    except ImportError:
        return {}
    try:
        return _finite_blob(_isinstance_floats(_core(seed=_BREEDEN_SEED)))
    except (ValueError, RuntimeError, FloatingPointError, KeyError, TypeError):
        return {}
