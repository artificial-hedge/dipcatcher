"""Benchmark batteries for SOTA canon wave 21 (see waves 11-20 for the pattern).

Wave 21 lands OPTIONAL research families, each pinned to the paper shipped
in its lane module (see the module docstrings for full citations; citation
ids verified against arXiv before implementation):

- ``bocpd_changepoint``: Bayesian online change-point detection
  (Adams & MacKay 2007, arXiv:0710.3742) — run-length posterior over a
  constant/geometric hazard, a Normal-Gamma conjugate learner emitting
  Student-t predictive scores, MAP-reset detected boundaries, and
  oracle/no-change comparator scores on planted multi-regime series.
  Pure numpy/scipy.
- ``rough_heston_rbergomi``: rough-volatility pricing machinery
  (El Euch & Rosenbaum 2019, arXiv:1609.02108; Bayer-Friz-Gatheral 2016;
  Abi Jaber-Larsson-Pulido 2019, arXiv:1708.08796) — the fractional
  Riccati characteristic function via Diethelm-Ford-Freed Adams-PC,
  Lewis-inversion call prices, Riemann-Liouville fBm weights, and
  correlated rBergomi / Volterra-Heston path simulators. Pure numpy/scipy.
- ``signature_features``: path-signature / signature-kernel metrics
  (Chevyrev & Oberhauser 2022, arXiv:1810.10971; Salvi, Cass, Foster,
  Lyons & Yang 2021, arXiv:2006.14794) — Goursat-PDE signature kernel
  validated against truncated inner products, lead-lag MMD two-sample
  power, log-signature drift regression. Composes with
  ``models/path_signatures`` (truncated sig/log-sig machinery); only the
  PDE kernel + batched Gram layer is new. Pure numpy/scipy.

Honesty contract: all blobs are SYNTHETIC correctness/telemetry checks, never
market evidence. No Sharpe/Sortino/Calmar/PnL/NAV tokens appear in emitted
keys; every bench returns ``{}`` (fails soft) if any emitted value is
non-finite — the ``_finite_blob`` gate, waves 15-20 precedent.

SHRUNK Monte-Carlo budgets (relative to each lane's own test suite):
module-default bench fixtures; lane modules report their own budgets.
"""

from __future__ import annotations

import math

_SEED = 20261126  # wave-21 stamp seed

# --- bocpd_changepoint: Bayesian online changepoint detection ----------------
_BOCPD_SEED = _SEED + 90

# --- rough_heston_rbergomi: fractional-Riccati CF + rBergomi ------------------
_RHRB_SEED = _SEED + 91

# --- signature_features: signature kernel MMD + log-sig drift ------------------
_SIG_SEED = _SEED + 92


def _finite_blob(mapped: dict[str, float]) -> dict[str, float]:
    """``{}`` unless every emitted value is finite (ruff-bench contract)."""
    if all(math.isfinite(v) for v in mapped.values()):
        return mapped
    return {}


def _isinstance_floats(raw: dict[str, float] | dict[str, object]) -> dict[str, float]:
    """Float-coerce a lane blob, dropping str stamps and runtime telemetry."""
    return {k: float(v) for k, v in raw.items() if isinstance(v, (int, float))}


def bench_bocpd_changepoint() -> dict[str, float]:
    """BOCPD detection quality + predictive-score bench (wave 21).

    Thin adapter over the lane module's own ``bench_bocpd_changepoint``:
    planted-regime detection delay/precision/recall, mean log-predictive
    vs oracle/no-change comparators. Returns ``{}`` while the lane module
    is absent or on fail-closed rejection.
    """
    try:
        from quant_fund.models.bocpd_changepoint import (
            bench_bocpd_changepoint as _bocpd_core_bench,
        )
    except ImportError:
        return {}
    try:
        return _finite_blob(_isinstance_floats(_bocpd_core_bench(seed=_BOCPD_SEED)))
    except (ValueError, RuntimeError, FloatingPointError, KeyError, TypeError):
        return {}


def bench_rough_heston_rbergomi() -> dict[str, float]:
    """Rough-volatility pricing diagnostics bench (wave 21).

    Thin adapter over the lane module's own ``bench_rough_heston_rbergomi``:
    fractional Riccati vs classical reduction, rBergomi leverage/Hurst
    signatures, CF-vs-MC ATM call gap, RL-fBm variance growth. Returns
    ``{}`` while the lane module is absent or on fail-closed rejection.
    """
    try:
        from quant_fund.models.rough_heston_rbergomi import (
            bench_rough_heston_rbergomi as _rhrb_core_bench,
        )
    except ImportError:
        return {}
    try:
        return _finite_blob(_isinstance_floats(_rhrb_core_bench(seed=_RHRB_SEED)))
    except (ValueError, RuntimeError, FloatingPointError, KeyError, TypeError):
        return {}


def bench_signature_features() -> dict[str, float]:
    """Signature-kernel diagnostics bench (wave 21).

    Thin adapter over the lane module's own ``bench_signature_features``:
    Goursat-PDE kernel vs truncated-signature inner product, MMD power
    separating GBM/OU/vol-shift ensembles, log-signature drift regression
    R^2, Gram-matrix sanity. Returns ``{}`` while the lane module is
    absent or on fail-closed rejection.
    """
    try:
        from quant_fund.metrics.signature_features import (
            bench_signature_features as _sig_core_bench,
        )
    except ImportError:
        return {}
    try:
        return _finite_blob(_isinstance_floats(_sig_core_bench(seed=_SIG_SEED)))
    except (ValueError, RuntimeError, FloatingPointError, KeyError, TypeError):
        return {}
