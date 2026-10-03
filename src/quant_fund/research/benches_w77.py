"""Wave-77 optional scorecard families — Liang-Zeger
generalized estimating equations, Laird-Ware linear
mixed models via EM ML, Cohen/Fleiss/Krippendorff
inter-rater agreement + Lin CCC + Bland-Altman,
Liu-Ting-Zhou isolation forest (+ Hariri extended
variant), HEGY seasonal unit roots + Canova-Hansen
seasonal stability, and van Buuren MICE chained
imputation + Rubin pooling.
Emitted only when the corresponding module's `bench_*` self-check
completes on its SYNTHETIC fixture.
"""

from __future__ import annotations

import math

_GEE_SEED = 20261231 + 450
_LMM_SEED = 20261231 + 451
_INTERRATER_SEED = 20261231 + 452
_ISOLATION_FOREST_SEED = 20261231 + 453
_HEGY_SEED = 20261231 + 454
_MICE_SEED = 20261231 + 455

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


def bench_gee() -> dict[str, float]:
    try:
        from quant_fund.models.gee import (
            bench_gee as _core,
        )
    except ImportError:  # pragma: no cover - optional
        return {}
    try:
        out = _core(seed=_GEE_SEED)
    except (ValueError, RuntimeError, FloatingPointError, KeyError, TypeError):
        return {}
    return _isinstance_floats(out)


def bench_lmm() -> dict[str, float]:
    try:
        from quant_fund.models.lmm import (
            bench_lmm as _core,
        )
    except ImportError:  # pragma: no cover - optional
        return {}
    try:
        out = _core(seed=_LMM_SEED)
    except (ValueError, RuntimeError, FloatingPointError, KeyError, TypeError):
        return {}
    return _isinstance_floats(out)


def bench_interrater() -> dict[str, float]:
    try:
        from quant_fund.models.interrater import (
            bench_interrater as _core,
        )
    except ImportError:  # pragma: no cover - optional
        return {}
    try:
        out = _core(seed=_INTERRATER_SEED)
    except (ValueError, RuntimeError, FloatingPointError, KeyError, TypeError):
        return {}
    return _isinstance_floats(out)


def bench_isolation_forest() -> dict[str, float]:
    try:
        from quant_fund.models.isolation_forest import bench_isolation_forest as _core
    except ImportError:  # pragma: no cover - optional
        return {}
    try:
        out = _core(seed=_ISOLATION_FOREST_SEED)
    except (ValueError, RuntimeError, FloatingPointError, KeyError, TypeError):
        return {}
    return _isinstance_floats(out)


def bench_hegy() -> dict[str, float]:
    try:
        from quant_fund.models.hegy import (
            bench_hegy as _core,
        )
    except ImportError:  # pragma: no cover - optional
        return {}
    try:
        out = _core(seed=_HEGY_SEED)
    except (ValueError, RuntimeError, FloatingPointError, KeyError, TypeError):
        return {}
    return _isinstance_floats(out)


def bench_mice() -> dict[str, float]:
    try:
        from quant_fund.models.mice import (
            bench_mice as _core,
        )
    except ImportError:  # pragma: no cover - optional
        return {}
    try:
        out = _core(seed=_MICE_SEED)
    except (ValueError, RuntimeError, FloatingPointError, KeyError, TypeError):
        return {}
    return _isinstance_floats(out)
