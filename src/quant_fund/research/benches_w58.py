"""Wave-58 optional benchmark adapters.

Each adapter runs one wave-58 module's self-check and returns a
``{metric_key: float}`` blob — all ``synthetic_*`` proper diagnostics, never
market evidence. A lane that is unavailable (ImportError) or whose numbers are
non-finite — the ``_finite_blob`` gate, waves 15-57 precedent — degrades to
``{}`` so the scorecard omits it cleanly.
"""

from __future__ import annotations

import math

_SEED = 20261231  # wave-58 stamp seed

_EXTREMOGRAM_SEED = _SEED + 336
_ECHO_STATE_SEED = _SEED + 337
_BATES_SVJ_SEED = _SEED + 338
_STL_LOESS_SEED = _SEED + 339
_PELT_WBS_SEED = _SEED + 340
_SPECTRAL_PCA_SEED = _SEED + 341

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


def bench_extremogram() -> dict[str, float]:
    try:
        from quant_fund.models.extremogram import bench_extremogram as _core
    except ImportError:
        return {}
    try:
        return _finite_blob(_isinstance_floats(_core(seed=_EXTREMOGRAM_SEED)))
    except (ValueError, RuntimeError, FloatingPointError, KeyError, TypeError):
        return {}


def bench_echo_state() -> dict[str, float]:
    try:
        from quant_fund.models.echo_state import bench_echo_state as _core
    except ImportError:
        return {}
    try:
        return _finite_blob(_isinstance_floats(_core(seed=_ECHO_STATE_SEED)))
    except (ValueError, RuntimeError, FloatingPointError, KeyError, TypeError):
        return {}


def bench_bates_svj() -> dict[str, float]:
    try:
        from quant_fund.models.bates_svj import bench_bates as _core
    except ImportError:
        return {}
    try:
        return _finite_blob(_isinstance_floats(_core(seed=_BATES_SVJ_SEED)))
    except (ValueError, RuntimeError, FloatingPointError, KeyError, TypeError):
        return {}


def bench_stl_loess() -> dict[str, float]:
    try:
        from quant_fund.models.stl_loess import bench_stl as _core
    except ImportError:
        return {}
    try:
        return _finite_blob(_isinstance_floats(_core(seed=_STL_LOESS_SEED)))
    except (ValueError, RuntimeError, FloatingPointError, KeyError, TypeError):
        return {}


def bench_pelt_wbs() -> dict[str, float]:
    try:
        from quant_fund.models.pelt_wbs import bench_pelt_wbs as _core
    except ImportError:
        return {}
    try:
        return _finite_blob(_isinstance_floats(_core(seed=_PELT_WBS_SEED)))
    except (ValueError, RuntimeError, FloatingPointError, KeyError, TypeError):
        return {}


def bench_spectral_pca() -> dict[str, float]:
    try:
        from quant_fund.models.spectral_pca import bench_spectral_pca as _core
    except ImportError:
        return {}
    try:
        return _finite_blob(_isinstance_floats(_core(seed=_SPECTRAL_PCA_SEED)))
    except (ValueError, RuntimeError, FloatingPointError, KeyError, TypeError):
        return {}
