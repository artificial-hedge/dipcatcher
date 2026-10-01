"""Wave-50 optional benchmark adapters.

Each adapter runs one wave-51 module's self-check and returns a
``{metric_key: float}`` blob — all ``synthetic_*`` proper diagnostics, never
market evidence. A lane that is unavailable (ImportError) or whose numbers are
non-finite — the ``_finite_blob`` gate, waves 15-50 precedent — degrades to
``{}`` so the scorecard omits it cleanly.
"""

from __future__ import annotations

import math

_SEED = 20261231  # wave-51 stamp seed

_BKM_MOMENTS_SEED = _SEED + 293
_GSADF_SEED = _SEED + 294
_PMG_ARDL_SEED = _SEED + 295
_ROSS_RECOVERY_SEED = _SEED + 296
_AIT_SAHALIA_SEED = _SEED + 297
_TODA_YAMAMOTO_SEED = _SEED + 298

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


def bench_bkm_moments() -> dict[str, float]:
    try:
        from quant_fund.models.bkm_moments import (
            bench_bkm_moments as _core,
        )
    except ImportError:
        return {}
    try:
        return _finite_blob(_isinstance_floats(_core(seed=_BKM_MOMENTS_SEED)))
    except (ValueError, RuntimeError, FloatingPointError, KeyError, TypeError):
        return {}


def bench_gsadf_bubble() -> dict[str, float]:
    try:
        from quant_fund.models.gsadf_bubble import (
            bench_gsadf as _core,
        )
    except ImportError:
        return {}
    try:
        return _finite_blob(_isinstance_floats(_core(seed=_GSADF_SEED)))
    except (ValueError, RuntimeError, FloatingPointError, KeyError, TypeError):
        return {}


def bench_pmg_ardl() -> dict[str, float]:
    try:
        from quant_fund.models.pmg_ardl import (
            bench_pmg_ardl as _core,
        )
    except ImportError:
        return {}
    try:
        return _finite_blob(_isinstance_floats(_core(seed=_PMG_ARDL_SEED)))
    except (ValueError, RuntimeError, FloatingPointError, KeyError, TypeError):
        return {}


def bench_ross_recovery() -> dict[str, float]:
    try:
        from quant_fund.models.ross_recovery import (
            bench_ross_recovery as _core,
        )
    except ImportError:
        return {}
    try:
        return _finite_blob(_isinstance_floats(_core(seed=_ROSS_RECOVERY_SEED)))
    except (ValueError, RuntimeError, FloatingPointError, KeyError, TypeError):
        return {}


def bench_ait_sahalia() -> dict[str, float]:
    try:
        from quant_fund.models.ait_sahalia import (
            bench_ait_sahalia as _core,
        )
    except ImportError:
        return {}
    try:
        return _finite_blob(_isinstance_floats(_core(seed=_AIT_SAHALIA_SEED)))
    except (ValueError, RuntimeError, FloatingPointError, KeyError, TypeError):
        return {}


def bench_toda_yamamoto() -> dict[str, float]:
    try:
        from quant_fund.models.toda_yamamoto import (
            bench_toda_yamamoto as _core,
        )
    except ImportError:
        return {}
    try:
        return _finite_blob(_isinstance_floats(_core(seed=_TODA_YAMAMOTO_SEED)))
    except (ValueError, RuntimeError, FloatingPointError, KeyError, TypeError):
        return {}
