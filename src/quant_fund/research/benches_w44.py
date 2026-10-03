"""Wave-44 optional benchmark adapters.

Each adapter runs one wave-44 module's self-check and returns a
``{metric_key: float}`` blob — all ``synthetic_*`` proper diagnostics, never
market evidence. A lane that is unavailable (ImportError) or whose numbers are
non-finite — the ``_finite_blob`` gate, waves 15-43 precedent — degrades to
``{}`` so the scorecard omits it cleanly.
"""

from __future__ import annotations

import math

_SEED = 20261231  # wave-44 stamp seed

# --- per-family seeds -------------------------------------------------------
_JOHANSEN_VECM_SEED = _SEED + 251
_PIN_MODEL_SEED = _SEED + 252
_KYLE_LAMBDA_SEED = _SEED + 253
_OSTER_BOUNDS_SEED = _SEED + 254
_STOREY_FDR_SEED = _SEED + 255
_KIEFER_VOGELSANG_SEED = _SEED + 256


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


def bench_johansen_vecm() -> dict[str, float]:
    try:
        from quant_fund.models.johansen_vecm import (
            bench_johansen_vecm as _core,
        )
    except ImportError:
        return {}
    try:
        return _finite_blob(_isinstance_floats(_core(seed=_JOHANSEN_VECM_SEED)))
    except (ValueError, RuntimeError, FloatingPointError, KeyError, TypeError):
        return {}


def bench_pin_model() -> dict[str, float]:
    try:
        from quant_fund.models.pin_model import (
            bench_pin_model as _core,
        )
    except ImportError:
        return {}
    try:
        return _finite_blob(_isinstance_floats(_core(seed=_PIN_MODEL_SEED)))
    except (ValueError, RuntimeError, FloatingPointError, KeyError, TypeError):
        return {}


def bench_kyle_lambda() -> dict[str, float]:
    try:
        from quant_fund.models.kyle_lambda import (
            bench_kyle_lambda as _core,
        )
    except ImportError:
        return {}
    try:
        return _finite_blob(_isinstance_floats(_core(seed=_KYLE_LAMBDA_SEED)))
    except (ValueError, RuntimeError, FloatingPointError, KeyError, TypeError):
        return {}


def bench_oster_bounds() -> dict[str, float]:
    try:
        from quant_fund.models.oster_bounds import (
            bench_oster_bounds as _core,
        )
    except ImportError:
        return {}
    try:
        return _finite_blob(_isinstance_floats(_core(seed=_OSTER_BOUNDS_SEED)))
    except (ValueError, RuntimeError, FloatingPointError, KeyError, TypeError):
        return {}


def bench_storey_fdr() -> dict[str, float]:
    try:
        from quant_fund.metrics.storey_fdr import (
            bench_storey_fdr as _core,
        )
    except ImportError:
        return {}
    try:
        return _finite_blob(_isinstance_floats(_core(seed=_STOREY_FDR_SEED)))
    except (ValueError, RuntimeError, FloatingPointError, KeyError, TypeError):
        return {}


def bench_kiefer_vogelsang() -> dict[str, float]:
    try:
        from quant_fund.metrics.kiefer_vogelsang import (
            bench_kiefer_vogelsang as _core,
        )
    except ImportError:
        return {}
    try:
        return _finite_blob(_isinstance_floats(_core(seed=_KIEFER_VOGELSANG_SEED)))
    except (ValueError, RuntimeError, FloatingPointError, KeyError, TypeError):
        return {}
