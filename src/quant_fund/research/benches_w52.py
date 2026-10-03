"""Wave-52 optional benchmark adapters.

Each adapter runs one wave-52 module's self-check and returns a
``{metric_key: float}`` blob — all ``synthetic_*`` proper diagnostics, never
market evidence. A lane that is unavailable (ImportError) or whose numbers are
non-finite — the ``_finite_blob`` gate, waves 15-51 precedent — degrades to
``{}`` so the scorecard omits it cleanly.
"""

from __future__ import annotations

import math

_SEED = 20261231  # wave-52 stamp seed

_NARDL_SEED = _SEED + 299
_GROWTH_AT_RISK_SEED = _SEED + 300
_MELICK_THOMAS_SEED = _SEED + 301
_BANDI_RUSSELL_SEED = _SEED + 302
_HONG_LI_SEED = _SEED + 303
_BEVERIDGE_NELSON_SEED = _SEED + 304

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


def bench_nardl() -> dict[str, float]:
    try:
        from quant_fund.models.nardl import (
            bench_nardl as _core,
        )
    except ImportError:
        return {}
    try:
        return _finite_blob(_isinstance_floats(_core(seed=_NARDL_SEED)))
    except (ValueError, RuntimeError, FloatingPointError, KeyError, TypeError):
        return {}


def bench_growth_at_risk() -> dict[str, float]:
    try:
        from quant_fund.models.growth_at_risk import (
            bench_growth_at_risk as _core,
        )
    except ImportError:
        return {}
    try:
        return _finite_blob(_isinstance_floats(_core(seed=_GROWTH_AT_RISK_SEED)))
    except (ValueError, RuntimeError, FloatingPointError, KeyError, TypeError):
        return {}


def bench_melick_thomas() -> dict[str, float]:
    try:
        from quant_fund.models.melick_thomas import (
            bench_melick_thomas as _core,
        )
    except ImportError:
        return {}
    try:
        return _finite_blob(_isinstance_floats(_core(seed=_MELICK_THOMAS_SEED)))
    except (ValueError, RuntimeError, FloatingPointError, KeyError, TypeError):
        return {}


def bench_bandi_russell() -> dict[str, float]:
    try:
        from quant_fund.models.bandi_russell import (
            bench_bandi_russell as _core,
        )
    except ImportError:
        return {}
    try:
        return _finite_blob(_isinstance_floats(_core(seed=_BANDI_RUSSELL_SEED)))
    except (ValueError, RuntimeError, FloatingPointError, KeyError, TypeError):
        return {}


def bench_hong_li() -> dict[str, float]:
    try:
        from quant_fund.models.hong_li import (
            bench_hong_li as _core,
        )
    except ImportError:
        return {}
    try:
        return _finite_blob(_isinstance_floats(_core(seed=_HONG_LI_SEED)))
    except (ValueError, RuntimeError, FloatingPointError, KeyError, TypeError):
        return {}


def bench_beveridge_nelson() -> dict[str, float]:
    try:
        from quant_fund.models.beveridge_nelson import (
            bench_beveridge_nelson as _core,
        )
    except ImportError:
        return {}
    try:
        return _finite_blob(_isinstance_floats(_core(seed=_BEVERIDGE_NELSON_SEED)))
    except (ValueError, RuntimeError, FloatingPointError, KeyError, TypeError):
        return {}
