"""Wave-49 optional benchmark adapters.

Each adapter runs one wave-49 module's self-check and returns a
``{metric_key: float}`` blob — all ``synthetic_*`` proper diagnostics, never
market evidence. A lane that is unavailable (ImportError) or whose numbers are
non-finite — the ``_finite_blob`` gate, waves 15-47 precedent — degrades to
``{}`` so the scorecard omits it cleanly.
"""

from __future__ import annotations

import math

_SEED = 20261231  # wave-49 stamp seed

_EISENBERG_NOE_SEED = _SEED + 281
_FIRE_SALES_SEED = _SEED + 282
_DELTA_COVAR_SEED = _SEED + 283
_BLANCHARD_QUAH_SEED = _SEED + 284
_TVP_VAR_SEED = _SEED + 285
_META_ANALYSIS_SEED = _SEED + 286

_FORBIDDEN_TOKENS = frozenset({"sharpe", "sortino", "calmar", "pnl", "nav"})


def _finite_blob(mapped: dict[str, float]) -> dict[str, float]:
    """``{}`` unless every emitted value is finite (ruff-bench contract)."""
    if all(math.isfinite(v) for v in mapped.values()):
        return mapped
    return {}


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


def bench_eisenberg_noe() -> dict[str, float]:
    try:
        from quant_fund.models.eisenberg_noe import (
            bench_eisenberg_noe as _core,
        )
    except ImportError:
        return {}
    try:
        return _finite_blob(_isinstance_floats(_core(seed=_EISENBERG_NOE_SEED)))
    except (ValueError, RuntimeError, FloatingPointError, KeyError, TypeError):
        return {}


def bench_fire_sales() -> dict[str, float]:
    try:
        from quant_fund.models.fire_sales import (
            bench_fire_sales as _core,
        )
    except ImportError:
        return {}
    try:
        return _finite_blob(_isinstance_floats(_core(seed=_FIRE_SALES_SEED)))
    except (ValueError, RuntimeError, FloatingPointError, KeyError, TypeError):
        return {}


def bench_delta_covar() -> dict[str, float]:
    try:
        from quant_fund.models.delta_covar import (
            bench_delta_covar as _core,
        )
    except ImportError:
        return {}
    try:
        return _finite_blob(_isinstance_floats(_core(seed=_DELTA_COVAR_SEED)))
    except (ValueError, RuntimeError, FloatingPointError, KeyError, TypeError):
        return {}


def bench_blanchard_quah() -> dict[str, float]:
    try:
        from quant_fund.models.blanchard_quah import (
            bench_blanchard_quah as _core,
        )
    except ImportError:
        return {}
    try:
        return _finite_blob(_isinstance_floats(_core(seed=_BLANCHARD_QUAH_SEED)))
    except (ValueError, RuntimeError, FloatingPointError, KeyError, TypeError):
        return {}


def bench_tvp_var() -> dict[str, float]:
    try:
        from quant_fund.models.tvp_var import (
            bench_tvp_var as _core,
        )
    except ImportError:
        return {}
    try:
        return _finite_blob(_isinstance_floats(_core(seed=_TVP_VAR_SEED)))
    except (ValueError, RuntimeError, FloatingPointError, KeyError, TypeError):
        return {}


def bench_meta_analysis() -> dict[str, float]:
    try:
        from quant_fund.models.meta_analysis import (
            bench_meta_analysis as _core,
        )
    except ImportError:
        return {}
    try:
        return _finite_blob(_isinstance_floats(_core(seed=_META_ANALYSIS_SEED)))
    except (ValueError, RuntimeError, FloatingPointError, KeyError, TypeError):
        return {}
