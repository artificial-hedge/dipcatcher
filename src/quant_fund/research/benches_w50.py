"""Wave-49 optional benchmark adapters.

Each adapter runs one wave-50 module's self-check and returns a
``{metric_key: float}`` blob — all ``synthetic_*`` proper diagnostics, never
market evidence. A lane that is unavailable (ImportError) or whose numbers are
non-finite — the ``_finite_blob`` gate, waves 15-49 precedent — degrades to
``{}`` so the scorecard omits it cleanly.
"""

from __future__ import annotations

import math

_SEED = 20261231  # wave-50 stamp seed

_ACD_DURATION_SEED = _SEED + 287
_HJM_SEED = _SEED + 288
_AFFINE_TERM_SEED = _SEED + 289
_GIL_PELAEZ_SEED = _SEED + 290
_HEDONIC_SEED = _SEED + 291
_DEA_SEED = _SEED + 292

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


def bench_acd_duration() -> dict[str, float]:
    try:
        from quant_fund.models.acd_duration import (
            bench_acd_duration as _core,
        )
    except ImportError:
        return {}
    try:
        return _finite_blob(_isinstance_floats(_core(seed=_ACD_DURATION_SEED)))
    except (ValueError, RuntimeError, FloatingPointError, KeyError, TypeError):
        return {}


def bench_hjm() -> dict[str, float]:
    try:
        from quant_fund.models.hjm import (
            bench_hjm as _core,
        )
    except ImportError:
        return {}
    try:
        return _finite_blob(_isinstance_floats(_core(seed=_HJM_SEED)))
    except (ValueError, RuntimeError, FloatingPointError, KeyError, TypeError):
        return {}


def bench_affine_term() -> dict[str, float]:
    try:
        from quant_fund.models.affine_term import (
            bench_affine_term as _core,
        )
    except ImportError:
        return {}
    try:
        return _finite_blob(_isinstance_floats(_core(seed=_AFFINE_TERM_SEED)))
    except (ValueError, RuntimeError, FloatingPointError, KeyError, TypeError):
        return {}


def bench_gil_pelaez() -> dict[str, float]:
    try:
        from quant_fund.models.gil_pelaez import (
            bench_gil_pelaez as _core,
        )
    except ImportError:
        return {}
    try:
        return _finite_blob(_isinstance_floats(_core(seed=_GIL_PELAEZ_SEED)))
    except (ValueError, RuntimeError, FloatingPointError, KeyError, TypeError):
        return {}


def bench_hedonic() -> dict[str, float]:
    try:
        from quant_fund.models.hedonic import (
            bench_hedonic as _core,
        )
    except ImportError:
        return {}
    try:
        return _finite_blob(_isinstance_floats(_core(seed=_HEDONIC_SEED)))
    except (ValueError, RuntimeError, FloatingPointError, KeyError, TypeError):
        return {}


def bench_dea() -> dict[str, float]:
    try:
        from quant_fund.models.dea import (
            bench_dea as _core,
        )
    except ImportError:
        return {}
    try:
        return _finite_blob(_isinstance_floats(_core(seed=_DEA_SEED)))
    except (ValueError, RuntimeError, FloatingPointError, KeyError, TypeError):
        return {}
