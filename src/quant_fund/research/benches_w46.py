"""Wave-46 optional benchmark adapters.

Each adapter runs one wave-46 module's self-check and returns a
``{metric_key: float}`` blob — all ``synthetic_*`` proper diagnostics, never
market evidence. A lane that is unavailable (ImportError) or whose numbers are
non-finite — the ``_finite_blob`` gate, waves 15-45 precedent — degrades to
``{}`` so the scorecard omits it cleanly.
"""

from __future__ import annotations

import math

_SEED = 20261231  # wave-46 stamp seed

_BLP_DEMAND_SEED = _SEED + 263
_OLLEY_PAKES_SEED = _SEED + 264
_RUST_DDC_SEED = _SEED + 265
_OAXACA_BLINDER_SEED = _SEED + 266
_BINSCATTER_SEED = _SEED + 267
_DFL_DECOMP_SEED = _SEED + 268

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


def bench_blp_demand() -> dict[str, float]:
    try:
        from quant_fund.models.blp_demand import (
            bench_blp_demand as _core,
        )
    except ImportError:
        return {}
    try:
        return _finite_blob(_isinstance_floats(_core(seed=_BLP_DEMAND_SEED)))
    except (ValueError, RuntimeError, FloatingPointError, KeyError, TypeError):
        return {}


def bench_olley_pakes() -> dict[str, float]:
    try:
        from quant_fund.models.olley_pakes import (
            bench_olley_pakes as _core,
        )
    except ImportError:
        return {}
    try:
        return _finite_blob(_isinstance_floats(_core(seed=_OLLEY_PAKES_SEED)))
    except (ValueError, RuntimeError, FloatingPointError, KeyError, TypeError):
        return {}


def bench_rust_ddc() -> dict[str, float]:
    try:
        from quant_fund.models.rust_ddc import (
            bench_rust_ddc as _core,
        )
    except ImportError:
        return {}
    try:
        return _finite_blob(_isinstance_floats(_core(seed=_RUST_DDC_SEED)))
    except (ValueError, RuntimeError, FloatingPointError, KeyError, TypeError):
        return {}


def bench_oaxaca_blinder() -> dict[str, float]:
    try:
        from quant_fund.models.oaxaca_blinder import (
            bench_oaxaca_blinder as _core,
        )
    except ImportError:
        return {}
    try:
        return _finite_blob(_isinstance_floats(_core(seed=_OAXACA_BLINDER_SEED)))
    except (ValueError, RuntimeError, FloatingPointError, KeyError, TypeError):
        return {}


def bench_binscatter() -> dict[str, float]:
    try:
        from quant_fund.models.binscatter import (
            bench_binscatter as _core,
        )
    except ImportError:
        return {}
    try:
        return _finite_blob(_isinstance_floats(_core(seed=_BINSCATTER_SEED)))
    except (ValueError, RuntimeError, FloatingPointError, KeyError, TypeError):
        return {}


def bench_dfl_decomp() -> dict[str, float]:
    try:
        from quant_fund.models.dfl_decomp import (
            bench_dfl_decomp as _core,
        )
    except ImportError:
        return {}
    try:
        return _finite_blob(_isinstance_floats(_core(seed=_DFL_DECOMP_SEED)))
    except (ValueError, RuntimeError, FloatingPointError, KeyError, TypeError):
        return {}
