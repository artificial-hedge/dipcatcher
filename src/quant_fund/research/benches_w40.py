"""Wave-40 optional benchmark adapters.

Each adapter runs one wave-40 module's self-check and returns a
``{metric_key: float}`` blob — all ``synthetic_*`` proper diagnostics, never
market evidence. A lane that is unavailable (ImportError) or whose numbers are
non-finite — the ``_finite_blob`` gate, waves 15-40 precedent — degrades to
``{}`` so the scorecard omits it cleanly.
"""

from __future__ import annotations

import math

_SEED = 20261231  # wave-40 stamp seed

# --- per-family seeds -------------------------------------------------------
_MIXED_LOGIT_SEED = _SEED + 227
_HURDLE_SEED = _SEED + 228
_SUR_MODEL_SEED = _SEED + 229
_CONNECTEDNESS_SEED = _SEED + 230
_NONPARAMETRIC_IV_SEED = _SEED + 231
_SUBSAMPLING_SEED = _SEED + 232


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


def bench_mixed_logit() -> dict[str, float]:
    try:
        from quant_fund.models.mixed_logit import (
            bench_mixed_logit as _core,
        )
    except ImportError:
        return {}
    try:
        return _finite_blob(_isinstance_floats(_core(seed=_MIXED_LOGIT_SEED)))
    except (ValueError, RuntimeError, FloatingPointError, KeyError, TypeError):
        return {}


def bench_hurdle() -> dict[str, float]:
    try:
        from quant_fund.models.hurdle import (
            bench_hurdle as _core,
        )
    except ImportError:
        return {}
    try:
        return _finite_blob(_isinstance_floats(_core(seed=_HURDLE_SEED)))
    except (ValueError, RuntimeError, FloatingPointError, KeyError, TypeError):
        return {}


def bench_sur_model() -> dict[str, float]:
    try:
        from quant_fund.models.sur_model import (
            bench_sur_model as _core,
        )
    except ImportError:
        return {}
    try:
        return _finite_blob(_isinstance_floats(_core(seed=_SUR_MODEL_SEED)))
    except (ValueError, RuntimeError, FloatingPointError, KeyError, TypeError):
        return {}


def bench_connectedness() -> dict[str, float]:
    try:
        from quant_fund.models.connectedness import bench_connectedness as _core
    except ImportError:
        return {}
    try:
        return _finite_blob(_isinstance_floats(_core(seed=_CONNECTEDNESS_SEED)))
    except (ValueError, RuntimeError, FloatingPointError, KeyError, TypeError):
        return {}


def bench_nonparametric_iv() -> dict[str, float]:
    try:
        from quant_fund.models.nonparametric_iv import (
            bench_nonparametric_iv as _core,
        )
    except ImportError:
        return {}
    try:
        return _finite_blob(_isinstance_floats(_core(seed=_NONPARAMETRIC_IV_SEED)))
    except (ValueError, RuntimeError, FloatingPointError, KeyError, TypeError):
        return {}


def bench_subsampling() -> dict[str, float]:
    try:
        from quant_fund.metrics.subsampling import (
            bench_subsampling as _core,
        )
    except ImportError:
        return {}
    try:
        return _finite_blob(_isinstance_floats(_core(seed=_SUBSAMPLING_SEED)))
    except (ValueError, RuntimeError, FloatingPointError, KeyError, TypeError):
        return {}
