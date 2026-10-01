"""Wave-36 optional benchmark adapters.

Each adapter runs one wave-36 module's self-check and returns a
``{metric_key: float}`` blob — all ``synthetic_*`` proper diagnostics, never
market evidence. A lane that is unavailable (ImportError) or whose numbers are
non-finite — the ``_finite_blob`` gate, waves 15-35 precedent — degrades to
``{}`` so the scorecard omits it cleanly.
"""

from __future__ import annotations

import math

_SEED = 20261231  # wave-36 stamp seed

# --- per-family seeds -------------------------------------------------------
_ARELLANO_BOND_SEED = _SEED + 203
_BVAR_MINNESOTA_SEED = _SEED + 204
_MEDIATION_ANALYSIS_SEED = _SEED + 205
_COMPETING_RISKS_SEED = _SEED + 206
_STOCHASTIC_FRONTIER_SEED = _SEED + 207
_REGRESSION_KINK_SEED = _SEED + 208


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


def bench_arellano_bond() -> dict[str, float]:
    try:
        from quant_fund.models.arellano_bond import bench_arellano_bond as _core
    except ImportError:
        return {}
    try:
        return _finite_blob(_isinstance_floats(_core(seed=_ARELLANO_BOND_SEED)))
    except (ValueError, RuntimeError, FloatingPointError, KeyError, TypeError):
        return {}


def bench_bvar_minnesota() -> dict[str, float]:
    try:
        from quant_fund.models.bvar_minnesota import bench_bvar_minnesota as _core
    except ImportError:
        return {}
    try:
        return _finite_blob(_isinstance_floats(_core(seed=_BVAR_MINNESOTA_SEED)))
    except (ValueError, RuntimeError, FloatingPointError, KeyError, TypeError):
        return {}


def bench_mediation_analysis() -> dict[str, float]:
    try:
        from quant_fund.models.mediation_analysis import (
            bench_mediation_analysis as _core,
        )
    except ImportError:
        return {}
    try:
        return _finite_blob(_isinstance_floats(_core(seed=_MEDIATION_ANALYSIS_SEED)))
    except (ValueError, RuntimeError, FloatingPointError, KeyError, TypeError):
        return {}


def bench_competing_risks() -> dict[str, float]:
    try:
        from quant_fund.models.competing_risks import bench_competing_risks as _core
    except ImportError:
        return {}
    try:
        return _finite_blob(_isinstance_floats(_core(seed=_COMPETING_RISKS_SEED)))
    except (ValueError, RuntimeError, FloatingPointError, KeyError, TypeError):
        return {}


def bench_stochastic_frontier() -> dict[str, float]:
    try:
        from quant_fund.models.stochastic_frontier import (
            bench_stochastic_frontier as _core,
        )
    except ImportError:
        return {}
    try:
        return _finite_blob(_isinstance_floats(_core(seed=_STOCHASTIC_FRONTIER_SEED)))
    except (ValueError, RuntimeError, FloatingPointError, KeyError, TypeError):
        return {}


def bench_regression_kink() -> dict[str, float]:
    try:
        from quant_fund.models.regression_kink import bench_regression_kink as _core
    except ImportError:
        return {}
    try:
        return _finite_blob(_isinstance_floats(_core(seed=_REGRESSION_KINK_SEED)))
    except (ValueError, RuntimeError, FloatingPointError, KeyError, TypeError):
        return {}
