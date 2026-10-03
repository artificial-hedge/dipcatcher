"""Wave-38 optional benchmark adapters.

Each adapter runs one wave-38 module's self-check and returns a
``{metric_key: float}`` blob — all ``synthetic_*`` proper diagnostics, never
market evidence. A lane that is unavailable (ImportError) or whose numbers are
non-finite — the ``_finite_blob`` gate, waves 15-37 precedent — degrades to
``{}`` so the scorecard omits it cleanly.
"""

from __future__ import annotations

import math

_SEED = 20261231  # wave-38 stamp seed

# --- per-family seeds -------------------------------------------------------
_CONTROL_FUNCTION_SEED = _SEED + 215
_KERNEL_REGRESSION_SEED = _SEED + 216
_CENSORED_QUANTILE_SEED = _SEED + 217
_THRESHOLD_AR_SEED = _SEED + 218
_FRACTIONAL_RESPONSE_SEED = _SEED + 219
_INTERVAL_CENSORING_SEED = _SEED + 220


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


def bench_control_function() -> dict[str, float]:
    try:
        from quant_fund.models.control_function import (
            bench_control_function as _core,
        )
    except ImportError:
        return {}
    try:
        return _finite_blob(_isinstance_floats(_core(seed=_CONTROL_FUNCTION_SEED)))
    except (ValueError, RuntimeError, FloatingPointError, KeyError, TypeError):
        return {}


def bench_kernel_regression() -> dict[str, float]:
    try:
        from quant_fund.models.kernel_regression import (
            bench_kernel_regression as _core,
        )
    except ImportError:
        return {}
    try:
        return _finite_blob(_isinstance_floats(_core(seed=_KERNEL_REGRESSION_SEED)))
    except (ValueError, RuntimeError, FloatingPointError, KeyError, TypeError):
        return {}


def bench_censored_quantile() -> dict[str, float]:
    try:
        from quant_fund.models.censored_quantile import (
            bench_censored_quantile as _core,
        )
    except ImportError:
        return {}
    try:
        return _finite_blob(_isinstance_floats(_core(seed=_CENSORED_QUANTILE_SEED)))
    except (ValueError, RuntimeError, FloatingPointError, KeyError, TypeError):
        return {}


def bench_threshold_ar() -> dict[str, float]:
    try:
        from quant_fund.models.threshold_ar import bench_threshold_ar as _core
    except ImportError:
        return {}
    try:
        return _finite_blob(_isinstance_floats(_core(seed=_THRESHOLD_AR_SEED)))
    except (ValueError, RuntimeError, FloatingPointError, KeyError, TypeError):
        return {}


def bench_fractional_response() -> dict[str, float]:
    try:
        from quant_fund.models.fractional_response import (
            bench_fractional_response as _core,
        )
    except ImportError:
        return {}
    try:
        return _finite_blob(_isinstance_floats(_core(seed=_FRACTIONAL_RESPONSE_SEED)))
    except (ValueError, RuntimeError, FloatingPointError, KeyError, TypeError):
        return {}


def bench_interval_censoring() -> dict[str, float]:
    try:
        from quant_fund.models.interval_censoring import (
            bench_interval_censoring as _core,
        )
    except ImportError:
        return {}
    try:
        return _finite_blob(_isinstance_floats(_core(seed=_INTERVAL_CENSORING_SEED)))
    except (ValueError, RuntimeError, FloatingPointError, KeyError, TypeError):
        return {}
