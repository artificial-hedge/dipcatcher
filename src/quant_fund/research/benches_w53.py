"""Wave-52 optional benchmark adapters.

Each adapter runs one wave-53 module's self-check and returns a
``{metric_key: float}`` blob — all ``synthetic_*`` proper diagnostics, never
market evidence. A lane that is unavailable (ImportError) or whose numbers are
non-finite — the ``_finite_blob`` gate, waves 15-52 precedent — degrades to
``{}`` so the scorecard omits it cleanly.
"""

from __future__ import annotations

import math

_SEED = 20261231  # wave-53 stamp seed

_CORRADI_SWANSON_SEED = _SEED + 305
_ENGLE_KRONER_BEKK_SEED = _SEED + 306
_HESTON_QE_SEED = _SEED + 307
_MODEL_CONFIDENCE_SET_SEED = _SEED + 308
_CHRISTOFFERSEN_PELLETIER_SEED = _SEED + 309
_SHEPPARD_HEAVY_SEED = _SEED + 310

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


def bench_corradi_swanson() -> dict[str, float]:
    try:
        from quant_fund.models.corradi_swanson import (
            bench_corradi_swanson as _core,
        )
    except ImportError:
        return {}
    try:
        return _finite_blob(_isinstance_floats(_core(seed=_CORRADI_SWANSON_SEED)))
    except (ValueError, RuntimeError, FloatingPointError, KeyError, TypeError):
        return {}


def bench_engle_kroner_bekk() -> dict[str, float]:
    try:
        from quant_fund.models.engle_kroner_bekk import (
            bench_engle_kroner_bekk as _core,
        )
    except ImportError:
        return {}
    try:
        return _finite_blob(_isinstance_floats(_core(seed=_ENGLE_KRONER_BEKK_SEED)))
    except (ValueError, RuntimeError, FloatingPointError, KeyError, TypeError):
        return {}


def bench_heston_qe() -> dict[str, float]:
    try:
        from quant_fund.models.heston_qe import (
            bench_heston_qe as _core,
        )
    except ImportError:
        return {}
    try:
        return _finite_blob(_isinstance_floats(_core(seed=_HESTON_QE_SEED)))
    except (ValueError, RuntimeError, FloatingPointError, KeyError, TypeError):
        return {}


def bench_model_confidence_set() -> dict[str, float]:
    try:
        from quant_fund.models.model_confidence_set import (
            bench_model_confidence_set as _core,
        )
    except ImportError:
        return {}
    try:
        return _finite_blob(_isinstance_floats(_core(seed=_MODEL_CONFIDENCE_SET_SEED)))
    except (ValueError, RuntimeError, FloatingPointError, KeyError, TypeError):
        return {}


def bench_christoffersen_pelletier() -> dict[str, float]:
    try:
        from quant_fund.models.christoffersen_pelletier import (
            bench_christoffersen_pelletier as _core,
        )
    except ImportError:
        return {}
    try:
        return _finite_blob(_isinstance_floats(_core(seed=_CHRISTOFFERSEN_PELLETIER_SEED)))
    except (ValueError, RuntimeError, FloatingPointError, KeyError, TypeError):
        return {}


def bench_sheppard_heavy() -> dict[str, float]:
    try:
        from quant_fund.models.sheppard_heavy import (
            bench_sheppard_heavy as _core,
        )
    except ImportError:
        return {}
    try:
        return _finite_blob(_isinstance_floats(_core(seed=_SHEPPARD_HEAVY_SEED)))
    except (ValueError, RuntimeError, FloatingPointError, KeyError, TypeError):
        return {}
