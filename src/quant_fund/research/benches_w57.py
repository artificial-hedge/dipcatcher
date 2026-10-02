"""Wave-57 optional benchmark adapters.

Each adapter runs one wave-57 module's self-check and returns a
``{metric_key: float}`` blob — all ``synthetic_*`` proper diagnostics, never
market evidence. A lane that is unavailable (ImportError) or whose numbers are
non-finite — the ``_finite_blob`` gate, waves 15-56 precedent — degrades to
``{}`` so the scorecard omits it cleanly.
"""

from __future__ import annotations

import math

_SEED = 20261231  # wave-57 stamp seed

_WAVELET_MODWT_SEED = _SEED + 330
_GEWEKE_SPECTRAL_SEED = _SEED + 331
_IVX_SEED = _SEED + 332
_BAI_NG_IC_SEED = _SEED + 333
_WOOLDRIDGE_SERIAL_SEED = _SEED + 334
_LASSO_PDS_SEED = _SEED + 335

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


def bench_wavelet_modwt() -> dict[str, float]:
    try:
        from quant_fund.models.wavelet_modwt import (
            bench_wavelet_modwt as _core,
        )
    except ImportError:
        return {}
    try:
        return _finite_blob(_isinstance_floats(_core(seed=_WAVELET_MODWT_SEED)))
    except (ValueError, RuntimeError, FloatingPointError, KeyError, TypeError):
        return {}


def bench_geweke_spectral() -> dict[str, float]:
    try:
        from quant_fund.models.geweke_spectral import (
            bench_geweke_spectral as _core,
        )
    except ImportError:
        return {}
    try:
        return _finite_blob(_isinstance_floats(_core(seed=_GEWEKE_SPECTRAL_SEED)))
    except (ValueError, RuntimeError, FloatingPointError, KeyError, TypeError):
        return {}


def bench_ivx() -> dict[str, float]:
    try:
        from quant_fund.models.ivx import bench_ivx as _core
    except ImportError:
        return {}
    try:
        return _finite_blob(_isinstance_floats(_core(seed=_IVX_SEED)))
    except (ValueError, RuntimeError, FloatingPointError, KeyError, TypeError):
        return {}


def bench_bai_ng_ic() -> dict[str, float]:
    try:
        from quant_fund.models.bai_ng_ic import bench_bai_ng as _core
    except ImportError:
        return {}
    try:
        return _finite_blob(_isinstance_floats(_core(seed=_BAI_NG_IC_SEED)))
    except (ValueError, RuntimeError, FloatingPointError, KeyError, TypeError):
        return {}


def bench_wooldridge_serial() -> dict[str, float]:
    try:
        from quant_fund.models.wooldridge_serial import (
            bench_wooldridge as _core,
        )
    except ImportError:
        return {}
    try:
        return _finite_blob(_isinstance_floats(_core(seed=_WOOLDRIDGE_SERIAL_SEED)))
    except (ValueError, RuntimeError, FloatingPointError, KeyError, TypeError):
        return {}


def bench_lasso_pds() -> dict[str, float]:
    try:
        from quant_fund.models.lasso_pds import bench_lasso_pds as _core
    except ImportError:
        return {}
    try:
        return _finite_blob(_isinstance_floats(_core(seed=_LASSO_PDS_SEED)))
    except (ValueError, RuntimeError, FloatingPointError, KeyError, TypeError):
        return {}
