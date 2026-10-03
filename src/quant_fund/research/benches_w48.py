"""Wave-46 optional benchmark adapters.

Each adapter runs one wave-48 module's self-check and returns a
``{metric_key: float}`` blob — all ``synthetic_*`` proper diagnostics, never
market evidence. A lane that is unavailable (ImportError) or whose numbers are
non-finite — the ``_finite_blob`` gate, waves 15-47 precedent — degrades to
``{}`` so the scorecard omits it cleanly.
"""

from __future__ import annotations

import math

_SEED = 20261231  # wave-48 stamp seed

_TMLE_SEED = _SEED + 275
_LEWBEL_IV_SEED = _SEED + 276
_PROXIMAL_CAUSAL_SEED = _SEED + 277
_COVER_UP_SEED = _SEED + 278
_VPIN_SEED = _SEED + 279
_MARGINAL_TREATMENT_SEED = _SEED + 280

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


def bench_tmle() -> dict[str, float]:
    try:
        from quant_fund.models.tmle import (
            bench_tmle as _core,
        )
    except ImportError:
        return {}
    try:
        return _finite_blob(_isinstance_floats(_core(seed=_TMLE_SEED)))
    except (ValueError, RuntimeError, FloatingPointError, KeyError, TypeError):
        return {}


def bench_lewbel_iv() -> dict[str, float]:
    try:
        from quant_fund.models.lewbel_iv import (
            bench_lewbel_iv as _core,
        )
    except ImportError:
        return {}
    try:
        return _finite_blob(_isinstance_floats(_core(seed=_LEWBEL_IV_SEED)))
    except (ValueError, RuntimeError, FloatingPointError, KeyError, TypeError):
        return {}


def bench_proximal_causal() -> dict[str, float]:
    try:
        from quant_fund.models.proximal_causal import (
            bench_proximal_causal as _core,
        )
    except ImportError:
        return {}
    try:
        return _finite_blob(_isinstance_floats(_core(seed=_PROXIMAL_CAUSAL_SEED)))
    except (ValueError, RuntimeError, FloatingPointError, KeyError, TypeError):
        return {}


def bench_cover_up() -> dict[str, float]:
    try:
        from quant_fund.models.cover_up import (
            bench_cover_up as _core,
        )
    except ImportError:
        return {}
    try:
        return _finite_blob(_isinstance_floats(_core(seed=_COVER_UP_SEED)))
    except (ValueError, RuntimeError, FloatingPointError, KeyError, TypeError):
        return {}


def bench_vpin() -> dict[str, float]:
    try:
        from quant_fund.models.vpin import (
            bench_vpin as _core,
        )
    except ImportError:
        return {}
    try:
        return _finite_blob(_isinstance_floats(_core(seed=_VPIN_SEED)))
    except (ValueError, RuntimeError, FloatingPointError, KeyError, TypeError):
        return {}


def bench_marginal_treatment() -> dict[str, float]:
    try:
        from quant_fund.models.marginal_treatment import (
            bench_marginal_treatment as _core,
        )
    except ImportError:
        return {}
    try:
        return _finite_blob(_isinstance_floats(_core(seed=_MARGINAL_TREATMENT_SEED)))
    except (ValueError, RuntimeError, FloatingPointError, KeyError, TypeError):
        return {}
