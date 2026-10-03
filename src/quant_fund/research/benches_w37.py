"""Wave-37 optional benchmark adapters.

Each adapter runs one wave-37 module's self-check and returns a
``{metric_key: float}`` blob — all ``synthetic_*`` proper diagnostics, never
market evidence. A lane that is unavailable (ImportError) or whose numbers are
non-finite — the ``_finite_blob`` gate, waves 15-36 precedent — degrades to
``{}`` so the scorecard omits it cleanly.
"""

from __future__ import annotations

import math

_SEED = 20261231  # wave-37 stamp seed

# --- per-family seeds -------------------------------------------------------
_SPATIAL_ECONOMETRICS_SEED = _SEED + 209
_ORDERED_CHOICE_SEED = _SEED + 210
_TRIPLE_DIFFERENCE_SEED = _SEED + 211
_DISTRIBUTION_REGRESSION_SEED = _SEED + 212
_SIMEX_SEED = _SEED + 213
_LP_DID_SEED = _SEED + 214


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


def bench_spatial_econometrics() -> dict[str, float]:
    try:
        from quant_fund.models.spatial_econometrics import (
            bench_spatial_econometrics as _core,
        )
    except ImportError:
        return {}
    try:
        return _finite_blob(_isinstance_floats(_core(seed=_SPATIAL_ECONOMETRICS_SEED)))
    except (ValueError, RuntimeError, FloatingPointError, KeyError, TypeError):
        return {}


def bench_ordered_choice() -> dict[str, float]:
    try:
        from quant_fund.models.ordered_choice import bench_ordered_choice as _core
    except ImportError:
        return {}
    try:
        return _finite_blob(_isinstance_floats(_core(seed=_ORDERED_CHOICE_SEED)))
    except (ValueError, RuntimeError, FloatingPointError, KeyError, TypeError):
        return {}


def bench_triple_difference() -> dict[str, float]:
    try:
        from quant_fund.models.triple_difference import (
            bench_triple_difference as _core,
        )
    except ImportError:
        return {}
    try:
        return _finite_blob(_isinstance_floats(_core(seed=_TRIPLE_DIFFERENCE_SEED)))
    except (ValueError, RuntimeError, FloatingPointError, KeyError, TypeError):
        return {}


def bench_distribution_regression() -> dict[str, float]:
    try:
        from quant_fund.models.distribution_regression import (
            bench_distribution_regression as _core,
        )
    except ImportError:
        return {}
    try:
        return _finite_blob(_isinstance_floats(_core(seed=_DISTRIBUTION_REGRESSION_SEED)))
    except (ValueError, RuntimeError, FloatingPointError, KeyError, TypeError):
        return {}


def bench_simex() -> dict[str, float]:
    try:
        from quant_fund.models.simex import bench_simex as _core
    except ImportError:
        return {}
    try:
        return _finite_blob(_isinstance_floats(_core(seed=_SIMEX_SEED)))
    except (ValueError, RuntimeError, FloatingPointError, KeyError, TypeError):
        return {}


def bench_lp_did() -> dict[str, float]:
    try:
        from quant_fund.models.lp_did import bench_lp_did as _core
    except ImportError:
        return {}
    try:
        return _finite_blob(_isinstance_floats(_core(seed=_LP_DID_SEED)))
    except (ValueError, RuntimeError, FloatingPointError, KeyError, TypeError):
        return {}
