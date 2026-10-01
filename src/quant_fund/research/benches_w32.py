"""Wave-32 optional benchmark adapters.

Each adapter runs one wave-32 module's self-check and returns a
``{metric_key: float}`` blob — all ``synthetic_*`` proper diagnostics, never
market evidence. A lane that is unavailable (ImportError) or whose numbers are
non-finite — the ``_finite_blob`` gate, waves 15-31 precedent — degrades to
``{}`` so the scorecard omits it cleanly.
"""

from __future__ import annotations

import math

_SEED = 20261231  # wave-32 stamp seed

# --- per-family seeds -------------------------------------------------------
_EXTREME_VALUE_SEED = _SEED + 179
_DOUBLE_ML_SEED = _SEED + 180
_BUNCHING_SEED = _SEED + 181
_CAUSAL_FOREST_SEED = _SEED + 182
_GAUSSIAN_PROCESS_SEED = _SEED + 183
_MARKOV_SWITCHING_SEED = _SEED + 184


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


def bench_extreme_value() -> dict[str, float]:
    try:
        from quant_fund.metrics.extreme_value import bench_extreme_value as _core
    except ImportError:
        return {}
    try:
        return _finite_blob(_isinstance_floats(_core(seed=_EXTREME_VALUE_SEED)))
    except (ValueError, RuntimeError, FloatingPointError, KeyError, TypeError):
        return {}


def bench_double_ml() -> dict[str, float]:
    try:
        from quant_fund.models.double_ml import bench_double_ml as _core
    except ImportError:
        return {}
    try:
        return _finite_blob(_isinstance_floats(_core(seed=_DOUBLE_ML_SEED)))
    except (ValueError, RuntimeError, FloatingPointError, KeyError, TypeError):
        return {}


def bench_bunching() -> dict[str, float]:
    try:
        from quant_fund.models.bunching import bench_bunching as _core
    except ImportError:
        return {}
    try:
        return _finite_blob(_isinstance_floats(_core(seed=_BUNCHING_SEED)))
    except (ValueError, RuntimeError, FloatingPointError, KeyError, TypeError):
        return {}


def bench_causal_forest() -> dict[str, float]:
    try:
        from quant_fund.models.causal_forest import bench_causal_forest as _core
    except ImportError:
        return {}
    try:
        return _finite_blob(_isinstance_floats(_core(seed=_CAUSAL_FOREST_SEED)))
    except (ValueError, RuntimeError, FloatingPointError, KeyError, TypeError):
        return {}


def bench_gaussian_process() -> dict[str, float]:
    try:
        from quant_fund.models.gaussian_process import bench_gaussian_process as _core
    except ImportError:
        return {}
    try:
        return _finite_blob(_isinstance_floats(_core(seed=_GAUSSIAN_PROCESS_SEED)))
    except (ValueError, RuntimeError, FloatingPointError, KeyError, TypeError):
        return {}


def bench_markov_switching() -> dict[str, float]:
    try:
        from quant_fund.models.markov_switching import bench_markov_switching as _core
    except ImportError:
        return {}
    try:
        return _finite_blob(_isinstance_floats(_core(seed=_MARKOV_SWITCHING_SEED)))
    except (ValueError, RuntimeError, FloatingPointError, KeyError, TypeError):
        return {}
