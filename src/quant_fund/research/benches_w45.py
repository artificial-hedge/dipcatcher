"""Wave-45 optional benchmark adapters.

Each adapter runs one wave-45 module's self-check and returns a
``{metric_key: float}`` blob — all ``synthetic_*`` proper diagnostics, never
market evidence. A lane that is unavailable (ImportError) or whose numbers are
non-finite — the ``_finite_blob`` gate, waves 15-44 precedent — degrades to
``{}`` so the scorecard omits it cleanly.
"""

from __future__ import annotations

import math

_SEED = 20261231  # wave-45 stamp seed

_CONLEY_SE_SEED = _SEED + 257
_DRISCOLL_KRAAY_SEED = _SEED + 258
_PESARAN_CCE_SEED = _SEED + 259
_WALD_SPRT_SEED = _SEED + 260
_LEE_BOUNDS_SEED = _SEED + 261
_BARRETT_DONALD_SEED = _SEED + 262

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


def bench_conley_se() -> dict[str, float]:
    try:
        from quant_fund.models.conley_se import (
            bench_conley_se as _core,
        )
    except ImportError:
        return {}
    try:
        return _finite_blob(_isinstance_floats(_core(seed=_CONLEY_SE_SEED)))
    except (ValueError, RuntimeError, FloatingPointError, KeyError, TypeError):
        return {}


def bench_driscoll_kraay() -> dict[str, float]:
    try:
        from quant_fund.models.driscoll_kraay import (
            bench_driscoll_kraay as _core,
        )
    except ImportError:
        return {}
    try:
        return _finite_blob(_isinstance_floats(_core(seed=_DRISCOLL_KRAAY_SEED)))
    except (ValueError, RuntimeError, FloatingPointError, KeyError, TypeError):
        return {}


def bench_pesaran_cce() -> dict[str, float]:
    try:
        from quant_fund.models.pesaran_cce import (
            bench_pesaran_cce as _core,
        )
    except ImportError:
        return {}
    try:
        return _finite_blob(_isinstance_floats(_core(seed=_PESARAN_CCE_SEED)))
    except (ValueError, RuntimeError, FloatingPointError, KeyError, TypeError):
        return {}


def bench_wald_sprt() -> dict[str, float]:
    try:
        from quant_fund.models.wald_sprt import (
            bench_wald_sprt as _core,
        )
    except ImportError:
        return {}
    try:
        return _finite_blob(_isinstance_floats(_core(seed=_WALD_SPRT_SEED)))
    except (ValueError, RuntimeError, FloatingPointError, KeyError, TypeError):
        return {}


def bench_lee_bounds() -> dict[str, float]:
    try:
        from quant_fund.models.lee_bounds import (
            bench_lee_bounds as _core,
        )
    except ImportError:
        return {}
    try:
        return _finite_blob(_isinstance_floats(_core(seed=_LEE_BOUNDS_SEED)))
    except (ValueError, RuntimeError, FloatingPointError, KeyError, TypeError):
        return {}


def bench_barrett_donald() -> dict[str, float]:
    try:
        from quant_fund.metrics.barrett_donald import (
            bench_barrett_donald as _core,
        )
    except ImportError:
        return {}
    try:
        return _finite_blob(_isinstance_floats(_core(seed=_BARRETT_DONALD_SEED)))
    except (ValueError, RuntimeError, FloatingPointError, KeyError, TypeError):
        return {}
