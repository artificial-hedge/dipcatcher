"""Wave-30 optional benchmark adapters.

Each adapter runs one wave-30 module's self-check and returns a
``{metric_key: float}`` blob — all ``synthetic_*`` proper diagnostics, never
market evidence. A lane that is unavailable (ImportError) or whose numbers are
non-finite — the ``_finite_blob`` gate, waves 15-29 precedent — degrades to
``{}`` so the scorecard omits it cleanly.
"""

from __future__ import annotations

import math

_SEED = 20261231  # wave-30 stamp seed

# --- per-family seeds -------------------------------------------------------
_LP_IV_SEED = _SEED + 167
_BISPECTRUM_SEED = _SEED + 168
_FUNCTIONAL_LINEAR_SEED = _SEED + 169
_GAS_SCORE_SEED = _SEED + 170
_COUNT_DATA_SEED = _SEED + 171
_LYAPUNOV_SEED = _SEED + 172


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


def bench_lp_iv() -> dict[str, float]:
    try:
        from quant_fund.models.lp_iv import bench_lp_iv as _core
    except ImportError:
        return {}
    try:
        return _finite_blob(_isinstance_floats(_core(seed=_LP_IV_SEED)))
    except (ValueError, RuntimeError, FloatingPointError, KeyError, TypeError):
        return {}


def bench_bispectrum() -> dict[str, float]:
    try:
        from quant_fund.metrics.bispectrum import bench_bispectrum as _core
    except ImportError:
        return {}
    try:
        return _finite_blob(_isinstance_floats(_core(seed=_BISPECTRUM_SEED)))
    except (ValueError, RuntimeError, FloatingPointError, KeyError, TypeError):
        return {}


def bench_functional_linear() -> dict[str, float]:
    try:
        from quant_fund.models.functional_linear import (
            bench_functional_linear as _core,
        )
    except ImportError:
        return {}
    try:
        return _finite_blob(_isinstance_floats(_core(seed=_FUNCTIONAL_LINEAR_SEED)))
    except (ValueError, RuntimeError, FloatingPointError, KeyError, TypeError):
        return {}


def bench_gas_score() -> dict[str, float]:
    try:
        from quant_fund.models.gas_score import bench_gas_score as _core
    except ImportError:
        return {}
    try:
        return _finite_blob(_isinstance_floats(_core(seed=_GAS_SCORE_SEED)))
    except (ValueError, RuntimeError, FloatingPointError, KeyError, TypeError):
        return {}


def bench_count_data() -> dict[str, float]:
    try:
        from quant_fund.models.count_data import bench_count_data as _core
    except ImportError:
        return {}
    try:
        return _finite_blob(_isinstance_floats(_core(seed=_COUNT_DATA_SEED)))
    except (ValueError, RuntimeError, FloatingPointError, KeyError, TypeError):
        return {}


def bench_lyapunov() -> dict[str, float]:
    try:
        from quant_fund.metrics.lyapunov import bench_lyapunov as _core
    except ImportError:
        return {}
    try:
        return _finite_blob(_isinstance_floats(_core(seed=_LYAPUNOV_SEED)))
    except (ValueError, RuntimeError, FloatingPointError, KeyError, TypeError):
        return {}
