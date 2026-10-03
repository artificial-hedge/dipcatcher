"""Wave-35 optional benchmark adapters.

Each adapter runs one wave-35 module's self-check and returns a
``{metric_key: float}`` blob — all ``synthetic_*`` proper diagnostics, never
market evidence. A lane that is unavailable (ImportError) or whose numbers are
non-finite — the ``_finite_blob`` gate, waves 15-34 precedent — degrades to
``{}`` so the scorecard omits it cleanly.
"""

from __future__ import annotations

import math

_SEED = 20261231  # wave-35 stamp seed

# --- per-family seeds -------------------------------------------------------
_HONEST_DID_SEED = _SEED + 197
_MANY_IV_SEED = _SEED + 198
_FAMA_MACBETH_SEED = _SEED + 199
_SPECIFICATION_CURVE_SEED = _SEED + 200
_SIGN_RESTRICTED_VAR_SEED = _SEED + 201
_PANEL_QUANTILE_FE_SEED = _SEED + 202


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


def bench_honest_did() -> dict[str, float]:
    try:
        from quant_fund.models.honest_did import bench_honest_did as _core
    except ImportError:
        return {}
    try:
        return _finite_blob(_isinstance_floats(_core(seed=_HONEST_DID_SEED)))
    except (ValueError, RuntimeError, FloatingPointError, KeyError, TypeError):
        return {}


def bench_many_iv() -> dict[str, float]:
    try:
        from quant_fund.models.many_iv import bench_many_iv as _core
    except ImportError:
        return {}
    try:
        return _finite_blob(_isinstance_floats(_core(seed=_MANY_IV_SEED)))
    except (ValueError, RuntimeError, FloatingPointError, KeyError, TypeError):
        return {}


def bench_fama_macbeth() -> dict[str, float]:
    try:
        from quant_fund.models.fama_macbeth import bench_fama_macbeth as _core
    except ImportError:
        return {}
    try:
        return _finite_blob(_isinstance_floats(_core(seed=_FAMA_MACBETH_SEED)))
    except (ValueError, RuntimeError, FloatingPointError, KeyError, TypeError):
        return {}


def bench_specification_curve() -> dict[str, float]:
    try:
        from quant_fund.models.specification_curve import (
            bench_specification_curve as _core,
        )
    except ImportError:
        return {}
    try:
        return _finite_blob(_isinstance_floats(_core(seed=_SPECIFICATION_CURVE_SEED)))
    except (ValueError, RuntimeError, FloatingPointError, KeyError, TypeError):
        return {}


def bench_sign_restricted_var() -> dict[str, float]:
    try:
        from quant_fund.models.sign_restricted_var import (
            bench_sign_restricted_var as _core,
        )
    except ImportError:
        return {}
    try:
        return _finite_blob(_isinstance_floats(_core(seed=_SIGN_RESTRICTED_VAR_SEED)))
    except (ValueError, RuntimeError, FloatingPointError, KeyError, TypeError):
        return {}


def bench_panel_quantile_fe() -> dict[str, float]:
    try:
        from quant_fund.models.panel_quantile_fe import (
            bench_panel_quantile_fe as _core,
        )
    except ImportError:
        return {}
    try:
        return _finite_blob(_isinstance_floats(_core(seed=_PANEL_QUANTILE_FE_SEED)))
    except (ValueError, RuntimeError, FloatingPointError, KeyError, TypeError):
        return {}
