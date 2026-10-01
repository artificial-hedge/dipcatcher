"""Wave-56 optional benchmark adapters.

Each adapter runs one wave-56 module's self-check and returns a
``{metric_key: float}`` blob — all ``synthetic_*`` proper diagnostics, never
market evidence. A lane that is unavailable (ImportError) or whose numbers are
non-finite — the ``_finite_blob`` gate, waves 15-55 precedent — degrades to
``{}`` so the scorecard omits it cleanly.
"""

from __future__ import annotations

import math

_SEED = 20261231  # wave-56 stamp seed

_KPSS_SEED = _SEED + 324
_DFGLS_SEED = _SEED + 325
_NG_PERRON_SEED = _SEED + 326
_PHILLIPS_PERRON_SEED = _SEED + 327
_ZIVOT_ANDREWS_SEED = _SEED + 328
_LEE_STRAZICICH_SEED = _SEED + 329

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


def bench_kpss() -> dict[str, float]:
    try:
        from quant_fund.models.kpss import bench_kpss as _core
    except ImportError:
        return {}
    try:
        return _finite_blob(_isinstance_floats(_core(seed=_KPSS_SEED)))
    except (ValueError, RuntimeError, FloatingPointError, KeyError, TypeError):
        return {}


def bench_ers_dfgls() -> dict[str, float]:
    try:
        from quant_fund.models.ers_dfgls import bench_dfgls as _core
    except ImportError:
        return {}
    try:
        return _finite_blob(_isinstance_floats(_core(seed=_DFGLS_SEED)))
    except (ValueError, RuntimeError, FloatingPointError, KeyError, TypeError):
        return {}


def bench_ng_perron() -> dict[str, float]:
    try:
        from quant_fund.models.ng_perron import bench_ng_perron as _core
    except ImportError:
        return {}
    try:
        return _finite_blob(_isinstance_floats(_core(seed=_NG_PERRON_SEED)))
    except (ValueError, RuntimeError, FloatingPointError, KeyError, TypeError):
        return {}


def bench_phillips_perron() -> dict[str, float]:
    try:
        from quant_fund.models.phillips_perron import (
            bench_phillips_perron as _core,
        )
    except ImportError:
        return {}
    try:
        return _finite_blob(_isinstance_floats(_core(seed=_PHILLIPS_PERRON_SEED)))
    except (ValueError, RuntimeError, FloatingPointError, KeyError, TypeError):
        return {}


def bench_zivot_andrews() -> dict[str, float]:
    try:
        from quant_fund.models.zivot_andrews import (
            bench_zivot_andrews as _core,
        )
    except ImportError:
        return {}
    try:
        return _finite_blob(_isinstance_floats(_core(seed=_ZIVOT_ANDREWS_SEED)))
    except (ValueError, RuntimeError, FloatingPointError, KeyError, TypeError):
        return {}


def bench_lee_strazicich() -> dict[str, float]:
    try:
        from quant_fund.models.lee_strazicich import (
            bench_lee_strazicich as _core,
        )
    except ImportError:
        return {}
    try:
        return _finite_blob(_isinstance_floats(_core(seed=_LEE_STRAZICICH_SEED)))
    except (ValueError, RuntimeError, FloatingPointError, KeyError, TypeError):
        return {}
