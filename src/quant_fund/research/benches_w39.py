"""Wave-38 optional benchmark adapters.

Each adapter runs one wave-39 module's self-check and returns a
``{metric_key: float}`` blob — all ``synthetic_*`` proper diagnostics, never
market evidence. A lane that is unavailable (ImportError) or whose numbers are
non-finite — the ``_finite_blob`` gate, waves 15-38 precedent — degrades to
``{}`` so the scorecard omits it cleanly.
"""

from __future__ import annotations

import math

_SEED = 20261231  # wave-39 stamp seed

# --- per-family seeds -------------------------------------------------------
_MAXIMUM_SCORE_SEED = _SEED + 221
_SIEVE_ESTIMATION_SEED = _SEED + 222
_NESTED_LOGIT_SEED = _SEED + 223
_AFT_MODEL_SEED = _SEED + 224
_DISTANCE_COVARIANCE_SEED = _SEED + 225
_PANEL_UNITROOT_SEED = _SEED + 226


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


def bench_maximum_score() -> dict[str, float]:
    try:
        from quant_fund.models.maximum_score import (
            bench_maximum_score as _core,
        )
    except ImportError:
        return {}
    try:
        return _finite_blob(_isinstance_floats(_core(seed=_MAXIMUM_SCORE_SEED)))
    except (ValueError, RuntimeError, FloatingPointError, KeyError, TypeError):
        return {}


def bench_sieve_estimation() -> dict[str, float]:
    try:
        from quant_fund.models.sieve_estimation import (
            bench_sieve_estimation as _core,
        )
    except ImportError:
        return {}
    try:
        return _finite_blob(_isinstance_floats(_core(seed=_SIEVE_ESTIMATION_SEED)))
    except (ValueError, RuntimeError, FloatingPointError, KeyError, TypeError):
        return {}


def bench_nested_logit() -> dict[str, float]:
    try:
        from quant_fund.models.nested_logit import (
            bench_nested_logit as _core,
        )
    except ImportError:
        return {}
    try:
        return _finite_blob(_isinstance_floats(_core(seed=_NESTED_LOGIT_SEED)))
    except (ValueError, RuntimeError, FloatingPointError, KeyError, TypeError):
        return {}


def bench_aft_model() -> dict[str, float]:
    try:
        from quant_fund.models.aft_model import bench_aft_model as _core
    except ImportError:
        return {}
    try:
        return _finite_blob(_isinstance_floats(_core(seed=_AFT_MODEL_SEED)))
    except (ValueError, RuntimeError, FloatingPointError, KeyError, TypeError):
        return {}


def bench_distance_covariance() -> dict[str, float]:
    try:
        from quant_fund.models.distance_covariance import (
            bench_distance_covariance as _core,
        )
    except ImportError:
        return {}
    try:
        return _finite_blob(_isinstance_floats(_core(seed=_DISTANCE_COVARIANCE_SEED)))
    except (ValueError, RuntimeError, FloatingPointError, KeyError, TypeError):
        return {}


def bench_panel_unitroot() -> dict[str, float]:
    try:
        from quant_fund.models.panel_unitroot import (
            bench_panel_unitroot as _core,
        )
    except ImportError:
        return {}
    try:
        return _finite_blob(_isinstance_floats(_core(seed=_PANEL_UNITROOT_SEED)))
    except (ValueError, RuntimeError, FloatingPointError, KeyError, TypeError):
        return {}
