"""Wave-31 optional benchmark adapters.

Each adapter runs one wave-31 module's self-check and returns a
``{metric_key: float}`` blob — all ``synthetic_*`` proper diagnostics, never
market evidence. A lane that is unavailable (ImportError) or whose numbers are
non-finite — the ``_finite_blob`` gate, waves 15-30 precedent — degrades to
``{}`` so the scorecard omits it cleanly.
"""

from __future__ import annotations

import math

_SEED = 20261231  # wave-31 stamp seed

# --- per-family seeds -------------------------------------------------------
_KERNEL_IV_SEED = _SEED + 173
_MULTISTATE_SEED = _SEED + 174
_PARTIAL_LINEAR_SEED = _SEED + 175
_HECKMAN_SEED = _SEED + 176
_RD_SEED = _SEED + 177
_BOUNDS_SEED = _SEED + 178


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


def bench_kernel_iv() -> dict[str, float]:
    try:
        from quant_fund.models.kernel_iv import bench_kernel_iv as _core
    except ImportError:
        return {}
    try:
        return _finite_blob(_isinstance_floats(_core(seed=_KERNEL_IV_SEED)))
    except (ValueError, RuntimeError, FloatingPointError, KeyError, TypeError):
        return {}


def bench_multistate() -> dict[str, float]:
    try:
        from quant_fund.models.multistate import bench_multistate as _core
    except ImportError:
        return {}
    try:
        return _finite_blob(_isinstance_floats(_core(seed=_MULTISTATE_SEED)))
    except (ValueError, RuntimeError, FloatingPointError, KeyError, TypeError):
        return {}


def bench_partial_linear() -> dict[str, float]:
    try:
        from quant_fund.models.partial_linear import bench_partial_linear as _core
    except ImportError:
        return {}
    try:
        return _finite_blob(_isinstance_floats(_core(seed=_PARTIAL_LINEAR_SEED)))
    except (ValueError, RuntimeError, FloatingPointError, KeyError, TypeError):
        return {}


def bench_heckman() -> dict[str, float]:
    try:
        from quant_fund.models.heckman import bench_heckman as _core
    except ImportError:
        return {}
    try:
        return _finite_blob(_isinstance_floats(_core(seed=_HECKMAN_SEED)))
    except (ValueError, RuntimeError, FloatingPointError, KeyError, TypeError):
        return {}


def bench_rd() -> dict[str, float]:
    try:
        from quant_fund.models.rd import bench_rd as _core
    except ImportError:
        return {}
    try:
        return _finite_blob(_isinstance_floats(_core(seed=_RD_SEED)))
    except (ValueError, RuntimeError, FloatingPointError, KeyError, TypeError):
        return {}


def bench_bounds() -> dict[str, float]:
    try:
        from quant_fund.models.bounds import bench_bounds as _core
    except ImportError:
        return {}
    try:
        return _finite_blob(_isinstance_floats(_core(seed=_BOUNDS_SEED)))
    except (ValueError, RuntimeError, FloatingPointError, KeyError, TypeError):
        return {}
