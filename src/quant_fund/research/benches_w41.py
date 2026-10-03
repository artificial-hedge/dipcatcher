"""Wave-41 optional benchmark adapters.

Each adapter runs one wave-41 module's self-check and returns a
``{metric_key: float}`` blob — all ``synthetic_*`` proper diagnostics, never
market evidence. A lane that is unavailable (ImportError) or whose numbers are
non-finite — the ``_finite_blob`` gate, waves 15-41 precedent — degrades to
``{}`` so the scorecard omits it cleanly.
"""

from __future__ import annotations

import math

_SEED = 20261231  # wave-41 stamp seed

# --- per-family seeds -------------------------------------------------------
_FRAILTY_SEED = _SEED + 233
_INTERRUPTED_TS_SEED = _SEED + 234
_LEAD_LAG_SEED = _SEED + 235
_PPML_SEED = _SEED + 236
_EVENT_STUDY_SEED = _SEED + 237
_MODEL_AVERAGING_SEED = _SEED + 238


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


def bench_frailty() -> dict[str, float]:
    try:
        from quant_fund.models.frailty import (
            bench_frailty as _core,
        )
    except ImportError:
        return {}
    try:
        return _finite_blob(_isinstance_floats(_core(seed=_FRAILTY_SEED)))
    except (ValueError, RuntimeError, FloatingPointError, KeyError, TypeError):
        return {}


def bench_interrupted_ts() -> dict[str, float]:
    try:
        from quant_fund.models.interrupted_ts import (
            bench_interrupted_ts as _core,
        )
    except ImportError:
        return {}
    try:
        return _finite_blob(_isinstance_floats(_core(seed=_INTERRUPTED_TS_SEED)))
    except (ValueError, RuntimeError, FloatingPointError, KeyError, TypeError):
        return {}


def bench_lead_lag() -> dict[str, float]:
    try:
        from quant_fund.metrics.lead_lag import (
            bench_lead_lag as _core,
        )
    except ImportError:
        return {}
    try:
        return _finite_blob(_isinstance_floats(_core(seed=_LEAD_LAG_SEED)))
    except (ValueError, RuntimeError, FloatingPointError, KeyError, TypeError):
        return {}


def bench_ppml() -> dict[str, float]:
    try:
        from quant_fund.models.ppml import bench_ppml as _core
    except ImportError:
        return {}
    try:
        return _finite_blob(_isinstance_floats(_core(seed=_PPML_SEED)))
    except (ValueError, RuntimeError, FloatingPointError, KeyError, TypeError):
        return {}


def bench_event_study() -> dict[str, float]:
    try:
        from quant_fund.metrics.event_study import (
            bench_event_study as _core,
        )
    except ImportError:
        return {}
    try:
        return _finite_blob(_isinstance_floats(_core(seed=_EVENT_STUDY_SEED)))
    except (ValueError, RuntimeError, FloatingPointError, KeyError, TypeError):
        return {}


def bench_model_averaging() -> dict[str, float]:
    try:
        from quant_fund.models.model_averaging import (
            bench_model_averaging as _core,
        )
    except ImportError:
        return {}
    try:
        return _finite_blob(_isinstance_floats(_core(seed=_MODEL_AVERAGING_SEED)))
    except (ValueError, RuntimeError, FloatingPointError, KeyError, TypeError):
        return {}
