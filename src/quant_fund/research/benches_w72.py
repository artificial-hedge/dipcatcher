"""Wave-72 optional scorecard families — exploratory
factor analysis, slice-sampling MCMC, rotation-invariant
CDF tests, penalized splines, scattered-data surfaces,
and shrinkage estimation.
Emitted only when the corresponding module's `bench_*` self-check
completes on its SYNTHETIC fixture.
"""

from __future__ import annotations

import math

_FACTOR_ANALYSIS_SEED = 20261231 + 420
_SLICE_SAMPLING_SEED = 20261231 + 421
_KUIPER_SEED = 20261231 + 422
_P_SPLINE_SEED = 20261231 + 423
_THIN_PLATE_SEED = 20261231 + 424
_JAMES_STEIN_SEED = 20261231 + 425

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}


def _finite_blob(mapped: dict[str, float]) -> dict[str, float]:
    """``{}`` unless every emitted value is finite (ruff-bench contract)."""
    if all(math.isfinite(v) for v in mapped.values()):
        return mapped
    return {}


def _isinstance_floats(raw: object) -> dict[str, float]:
    """Float-coerce a lane blob, dropping non-numeric entries and
    any key carrying a forbidden headline metric token."""
    if not isinstance(raw, dict):
        return {}
    return _finite_blob(
        {
            k: float(v)
            for k, v in raw.items()
            if isinstance(v, (int, float))
            and not isinstance(v, bool)
            and _FORBIDDEN.isdisjoint(k.lower().split("_"))
        }
    )


def bench_factor_analysis() -> dict[str, float]:
    try:
        from quant_fund.models.factor_analysis import (
            bench_factor_analysis as _core,
        )
    except ImportError:  # pragma: no cover - optional
        return {}
    try:
        out = _core(seed=_FACTOR_ANALYSIS_SEED)
    except (ValueError, RuntimeError, FloatingPointError, KeyError, TypeError):
        return {}
    return _isinstance_floats(out)


def bench_slice_sampling() -> dict[str, float]:
    try:
        from quant_fund.models.slice_sampling import (
            bench_slice_sampling as _core,
        )
    except ImportError:  # pragma: no cover - optional
        return {}
    try:
        out = _core(seed=_SLICE_SAMPLING_SEED)
    except (ValueError, RuntimeError, FloatingPointError, KeyError, TypeError):
        return {}
    return _isinstance_floats(out)


def bench_kuiper() -> dict[str, float]:
    try:
        from quant_fund.models.kuiper import (
            bench_kuiper as _core,
        )
    except ImportError:  # pragma: no cover - optional
        return {}
    try:
        out = _core(seed=_KUIPER_SEED)
    except (ValueError, RuntimeError, FloatingPointError, KeyError, TypeError):
        return {}
    return _isinstance_floats(out)


def bench_p_spline() -> dict[str, float]:
    try:
        from quant_fund.models.p_spline import bench_p_spline as _core
    except ImportError:  # pragma: no cover - optional
        return {}
    try:
        out = _core(seed=_P_SPLINE_SEED)
    except (ValueError, RuntimeError, FloatingPointError, KeyError, TypeError):
        return {}
    return _isinstance_floats(out)


def bench_thin_plate() -> dict[str, float]:
    try:
        from quant_fund.models.thin_plate import (
            bench_thin_plate as _core,
        )
    except ImportError:  # pragma: no cover - optional
        return {}
    try:
        out = _core(seed=_THIN_PLATE_SEED)
    except (ValueError, RuntimeError, FloatingPointError, KeyError, TypeError):
        return {}
    return _isinstance_floats(out)


def bench_james_stein() -> dict[str, float]:
    try:
        from quant_fund.models.james_stein import (
            bench_james_stein as _core,
        )
    except ImportError:  # pragma: no cover - optional
        return {}
    try:
        out = _core(seed=_JAMES_STEIN_SEED)
    except (ValueError, RuntimeError, FloatingPointError, KeyError, TypeError):
        return {}
    return _isinstance_floats(out)
