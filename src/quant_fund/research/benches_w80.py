"""Wave-80 optional scorecard families — Royston-Parmar
restricted mean survival time with Uno variance and
two-arm z-tests, Hull OIS zero-curve bootstrap with
log-linear discount interpolation and fixed-point
tenor stripping, Henze-Zirkler BHEP multivariate
normality with lognormal null, Epps-Singleton empirical-
characteristic-function normality, Watson U^2 circular
uniformity with Stephens critical table, and
Szekely-Rizzo energy-distance MVN test with parametric
bootstrap p-values.
Emitted only when the corresponding module's `bench_*` self-check
completes on its SYNTHETIC fixture.
"""

from __future__ import annotations

import math

_RMST_SEED = 20261231 + 468
_OIS_CURVE_SEED = 20261231 + 469
_HENZE_ZIRKLER_SEED = 20261231 + 470
_EPPS_SINGLETON_SEED = 20261231 + 471
_WATSON_SEED = 20261231 + 472
_ENERGY_TEST_SEED = 20261231 + 473

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


def bench_rmst() -> dict[str, float]:
    try:
        from quant_fund.models.rmst import bench_rmst as _core
    except ImportError:  # pragma: no cover - optional
        return {}
    try:
        out = _core(seed=_RMST_SEED)
    except (ValueError, RuntimeError, FloatingPointError, KeyError, TypeError):
        return {}
    return _isinstance_floats(out)


def bench_ois_curve() -> dict[str, float]:
    try:
        from quant_fund.models.ois_curve import bench_ois_curve as _core
    except ImportError:  # pragma: no cover - optional
        return {}
    try:
        out = _core(seed=_OIS_CURVE_SEED)
    except (ValueError, RuntimeError, FloatingPointError, KeyError, TypeError):
        return {}
    return _isinstance_floats(out)


def bench_henze_zirkler() -> dict[str, float]:
    try:
        from quant_fund.models.henze_zirkler import (
            bench_henze_zirkler as _core,
        )
    except ImportError:  # pragma: no cover - optional
        return {}
    try:
        out = _core(seed=_HENZE_ZIRKLER_SEED)
    except (ValueError, RuntimeError, FloatingPointError, KeyError, TypeError):
        return {}
    return _isinstance_floats(out)


def bench_epps_singleton() -> dict[str, float]:
    try:
        from quant_fund.models.epps_singleton import (
            bench_epps_singleton as _core,
        )
    except ImportError:  # pragma: no cover - optional
        return {}
    try:
        out = _core(seed=_EPPS_SINGLETON_SEED)
    except (ValueError, RuntimeError, FloatingPointError, KeyError, TypeError):
        return {}
    return _isinstance_floats(out)


def bench_watson() -> dict[str, float]:
    try:
        from quant_fund.models.watson import bench_watson as _core
    except ImportError:  # pragma: no cover - optional
        return {}
    try:
        out = _core(seed=_WATSON_SEED)
    except (ValueError, RuntimeError, FloatingPointError, KeyError, TypeError):
        return {}
    return _isinstance_floats(out)


def bench_energy_test() -> dict[str, float]:
    try:
        from quant_fund.models.energy_test import bench_energy_test as _core
    except ImportError:  # pragma: no cover - optional
        return {}
    try:
        out = _core(seed=_ENERGY_TEST_SEED)
    except (ValueError, RuntimeError, FloatingPointError, KeyError, TypeError):
        return {}
    return _isinstance_floats(out)
