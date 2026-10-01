"""Wave-75 optional scorecard families — Cronbach
alpha / KR-20 / split-half reliability, McGraw-Wong
ICC forms, Mantel-Haenszel + logistic DIF, G-theory
variance components, McDonald's omega, and Rasch
infit/outfit.
Emitted only when the corresponding module's `bench_*` self-check
completes on its SYNTHETIC fixture.
"""

from __future__ import annotations

import math

_CRONBACH_SEED = 20261231 + 438
_ICC_SEED = 20261231 + 439
_DIF_SEED = 20261231 + 440
_G_THEORY_SEED = 20261231 + 441
_OMEGA_SEED = 20261231 + 442
_RASCH_FIT_SEED = 20261231 + 443

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


def bench_cronbach() -> dict[str, float]:
    try:
        from quant_fund.models.cronbach import (
            bench_cronbach as _core,
        )
    except ImportError:  # pragma: no cover - optional
        return {}
    try:
        out = _core(seed=_CRONBACH_SEED)
    except (ValueError, RuntimeError, FloatingPointError, KeyError, TypeError):
        return {}
    return _isinstance_floats(out)


def bench_icc() -> dict[str, float]:
    try:
        from quant_fund.models.icc import (
            bench_icc as _core,
        )
    except ImportError:  # pragma: no cover - optional
        return {}
    try:
        out = _core(seed=_ICC_SEED)
    except (ValueError, RuntimeError, FloatingPointError, KeyError, TypeError):
        return {}
    return _isinstance_floats(out)


def bench_dif() -> dict[str, float]:
    try:
        from quant_fund.models.dif import (
            bench_dif as _core,
        )
    except ImportError:  # pragma: no cover - optional
        return {}
    try:
        out = _core(seed=_DIF_SEED)
    except (ValueError, RuntimeError, FloatingPointError, KeyError, TypeError):
        return {}
    return _isinstance_floats(out)


def bench_g_theory() -> dict[str, float]:
    try:
        from quant_fund.models.g_theory import bench_g_theory as _core
    except ImportError:  # pragma: no cover - optional
        return {}
    try:
        out = _core(seed=_G_THEORY_SEED)
    except (ValueError, RuntimeError, FloatingPointError, KeyError, TypeError):
        return {}
    return _isinstance_floats(out)


def bench_omega() -> dict[str, float]:
    try:
        from quant_fund.models.omega import (
            bench_omega as _core,
        )
    except ImportError:  # pragma: no cover - optional
        return {}
    try:
        out = _core(seed=_OMEGA_SEED)
    except (ValueError, RuntimeError, FloatingPointError, KeyError, TypeError):
        return {}
    return _isinstance_floats(out)


def bench_rasch_fit() -> dict[str, float]:
    try:
        from quant_fund.models.rasch_fit import (
            bench_rasch_fit as _core,
        )
    except ImportError:  # pragma: no cover - optional
        return {}
    try:
        out = _core(seed=_RASCH_FIT_SEED)
    except (ValueError, RuntimeError, FloatingPointError, KeyError, TypeError):
        return {}
    return _isinstance_floats(out)
