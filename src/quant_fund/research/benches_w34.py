"""Wave-34 optional benchmark adapters.

Each adapter runs one wave-34 module's self-check and returns a
``{metric_key: float}`` blob — all ``synthetic_*`` proper diagnostics, never
market evidence. A lane that is unavailable (ImportError) or whose numbers are
non-finite — the ``_finite_blob`` gate, waves 15-33 precedent — degrades to
``{}`` so the scorecard omits it cleanly.
"""

from __future__ import annotations

import math

_SEED = 20261231  # wave-34 stamp seed

# --- per-family seeds -------------------------------------------------------
_MATRIX_COMPLETION_SEED = _SEED + 191
_GSYNTH_SEED = _SEED + 192
_RIF_REGRESSION_SEED = _SEED + 193
_SHIFT_SHARE_SEED = _SEED + 194
_ENTROPY_BALANCING_SEED = _SEED + 195
_DID_DIAGNOSTICS_SEED = _SEED + 196


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


def bench_matrix_completion() -> dict[str, float]:
    try:
        from quant_fund.models.matrix_completion import bench_matrix_completion as _core
    except ImportError:
        return {}
    try:
        return _finite_blob(_isinstance_floats(_core(seed=_MATRIX_COMPLETION_SEED)))
    except (ValueError, RuntimeError, FloatingPointError, KeyError, TypeError):
        return {}


def bench_gsynth() -> dict[str, float]:
    try:
        from quant_fund.models.gsynth import bench_gsynth as _core
    except ImportError:
        return {}
    try:
        return _finite_blob(_isinstance_floats(_core(seed=_GSYNTH_SEED)))
    except (ValueError, RuntimeError, FloatingPointError, KeyError, TypeError):
        return {}


def bench_rif_regression() -> dict[str, float]:
    try:
        from quant_fund.models.rif_regression import bench_rif_regression as _core
    except ImportError:
        return {}
    try:
        return _finite_blob(_isinstance_floats(_core(seed=_RIF_REGRESSION_SEED)))
    except (ValueError, RuntimeError, FloatingPointError, KeyError, TypeError):
        return {}


def bench_shift_share() -> dict[str, float]:
    try:
        from quant_fund.models.shift_share import bench_shift_share as _core
    except ImportError:
        return {}
    try:
        return _finite_blob(_isinstance_floats(_core(seed=_SHIFT_SHARE_SEED)))
    except (ValueError, RuntimeError, FloatingPointError, KeyError, TypeError):
        return {}


def bench_entropy_balancing() -> dict[str, float]:
    try:
        from quant_fund.models.entropy_balancing import bench_entropy_balancing as _core
    except ImportError:
        return {}
    try:
        return _finite_blob(_isinstance_floats(_core(seed=_ENTROPY_BALANCING_SEED)))
    except (ValueError, RuntimeError, FloatingPointError, KeyError, TypeError):
        return {}


def bench_did_diagnostics() -> dict[str, float]:
    try:
        from quant_fund.models.did_diagnostics import bench_did_diagnostics as _core
    except ImportError:
        return {}
    try:
        return _finite_blob(_isinstance_floats(_core(seed=_DID_DIAGNOSTICS_SEED)))
    except (ValueError, RuntimeError, FloatingPointError, KeyError, TypeError):
        return {}
