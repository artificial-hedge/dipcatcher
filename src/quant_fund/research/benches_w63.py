"""Wave-63 adapter benches — optional scorecard families.

Each adapter calls a SYNTHETIC bench on the wave-63 models and
returns a small metric dict consumed by the scorecard runner.
Adapters are honest by contract: they propagate ``score`` and
label every metric ``synthetic_*``. Any import/shape/exception
failure returns ``{}`` so the family is skipped, never faked.

Families (seeds 20261231 + 366..371):
- ``higham_corr``     — Higham/Qi-Sun nearest correlation matrix
- ``lee_carter``      — Lee-Carter stochastic mortality (SVD + RW kappa)
- ``power_law``       — Clauset-Shalizi-Newman power-law tail fit
- ``convexity_adj``   — Vasicek/Hull-White futures convexity adjustment
- ``jln_uncertainty`` — Jurado-Ludvigson-Ng macro uncertainty factor
- ``first_passage``   — inverse-Gaussian hitting time + Siegmund lift
"""

from __future__ import annotations

import math

_SEED = 20261231
_HIGHAM_CORR_SEED = _SEED + 366
_LEE_CARTER_SEED = _SEED + 367
_POWER_LAW_SEED = _SEED + 368
_CONVEXITY_ADJ_SEED = _SEED + 369
_JLN_UNCERTAINTY_SEED = _SEED + 370
_FIRST_PASSAGE_SEED = _SEED + 371

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


def bench_higham_corr() -> dict[str, float]:
    try:
        from quant_fund.models.higham_corr import bench_higham as _core
    except ImportError:  # pragma: no cover - optional
        return {}
    try:
        out = _core(seed=_HIGHAM_CORR_SEED)
    except (ValueError, RuntimeError, FloatingPointError, KeyError, TypeError):
        return {}
    return _isinstance_floats(out)


def bench_lee_carter() -> dict[str, float]:
    try:
        from quant_fund.models.lee_carter import bench_lee_carter as _core
    except ImportError:  # pragma: no cover - optional
        return {}
    try:
        out = _core(seed=_LEE_CARTER_SEED)
    except (ValueError, RuntimeError, FloatingPointError, KeyError, TypeError):
        return {}
    return _isinstance_floats(out)


def bench_power_law() -> dict[str, float]:
    try:
        from quant_fund.models.power_law import bench_powerlaw as _core
    except ImportError:  # pragma: no cover - optional
        return {}
    try:
        out = _core(seed=_POWER_LAW_SEED)
    except (ValueError, RuntimeError, FloatingPointError, KeyError, TypeError):
        return {}
    return _isinstance_floats(out)


def bench_convexity_adj() -> dict[str, float]:
    try:
        from quant_fund.models.convexity_adj import bench_convexity as _core
    except ImportError:  # pragma: no cover - optional
        return {}
    try:
        out = _core(seed=_CONVEXITY_ADJ_SEED)
    except (ValueError, RuntimeError, FloatingPointError, KeyError, TypeError):
        return {}
    return _isinstance_floats(out)


def bench_jln_uncertainty() -> dict[str, float]:
    try:
        from quant_fund.models.jln_uncertainty import bench_jln as _core
    except ImportError:  # pragma: no cover - optional
        return {}
    try:
        out = _core(seed=_JLN_UNCERTAINTY_SEED)
    except (ValueError, RuntimeError, FloatingPointError, KeyError, TypeError):
        return {}
    return _isinstance_floats(out)


def bench_first_passage() -> dict[str, float]:
    try:
        from quant_fund.models.first_passage import (
            bench_first_passage as _core,
        )
    except ImportError:  # pragma: no cover - optional
        return {}
    try:
        out = _core(seed=_FIRST_PASSAGE_SEED)
    except (ValueError, RuntimeError, FloatingPointError, KeyError, TypeError):
        return {}
    return _isinstance_floats(out)
