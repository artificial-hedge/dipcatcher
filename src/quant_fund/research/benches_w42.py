"""Wave-42 optional benchmark adapters.

Each adapter runs one wave-42 module's self-check and returns a
``{metric_key: float}`` blob — all ``synthetic_*`` proper diagnostics, never
market evidence. A lane that is unavailable (ImportError) or whose numbers are
non-finite — the ``_finite_blob`` gate, waves 15-41 precedent — degrades to
``{}`` so the scorecard omits it cleanly.
"""

from __future__ import annotations

import math

_SEED = 20261231  # wave-42 stamp seed

# --- per-family seeds -------------------------------------------------------
_VARIANCE_RATIO_SEED = _SEED + 239
_HAR_RV_SEED = _SEED + 240
_CLARK_WEST_SEED = _SEED + 241
_STAMBAUGH_SEED = _SEED + 242
_ROY_MODEL_SEED = _SEED + 243
_TWO_WAY_CLUSTER_SEED = _SEED + 244


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


def bench_variance_ratio() -> dict[str, float]:
    try:
        from quant_fund.metrics.variance_ratio import (
            bench_variance_ratio as _core,
        )
    except ImportError:
        return {}
    try:
        return _finite_blob(_isinstance_floats(_core(seed=_VARIANCE_RATIO_SEED)))
    except (ValueError, RuntimeError, FloatingPointError, KeyError, TypeError):
        return {}


def bench_har_rv() -> dict[str, float]:
    try:
        from quant_fund.models.har_rv import (
            bench_har_rv as _core,
        )
    except ImportError:
        return {}
    try:
        return _finite_blob(_isinstance_floats(_core(seed=_HAR_RV_SEED)))
    except (ValueError, RuntimeError, FloatingPointError, KeyError, TypeError):
        return {}


def bench_clark_west() -> dict[str, float]:
    try:
        from quant_fund.models.clark_west import (
            bench_clark_west as _core,
        )
    except ImportError:
        return {}
    try:
        return _finite_blob(_isinstance_floats(_core(seed=_CLARK_WEST_SEED)))
    except (ValueError, RuntimeError, FloatingPointError, KeyError, TypeError):
        return {}


def bench_stambaugh() -> dict[str, float]:
    try:
        from quant_fund.models.stambaugh import bench_stambaugh as _core
    except ImportError:
        return {}
    try:
        return _finite_blob(_isinstance_floats(_core(seed=_STAMBAUGH_SEED)))
    except (ValueError, RuntimeError, FloatingPointError, KeyError, TypeError):
        return {}


def bench_roy_model() -> dict[str, float]:
    try:
        from quant_fund.models.roy_model import (
            bench_roy_model as _core,
        )
    except ImportError:
        return {}
    try:
        return _finite_blob(_isinstance_floats(_core(seed=_ROY_MODEL_SEED)))
    except (ValueError, RuntimeError, FloatingPointError, KeyError, TypeError):
        return {}


def bench_two_way_cluster() -> dict[str, float]:
    try:
        from quant_fund.metrics.two_way_cluster import (
            bench_two_way_cluster as _core,
        )
    except ImportError:
        return {}
    try:
        return _finite_blob(_isinstance_floats(_core(seed=_TWO_WAY_CLUSTER_SEED)))
    except (ValueError, RuntimeError, FloatingPointError, KeyError, TypeError):
        return {}
