"""Wave-73 optional scorecard families — multivariate
normality (Mardia), distance-matrix (Mantel), spatial
autocorrelation (Moran/Geary), blocked rank tests
(Friedman/Page), contingency tables (Fisher/McNemar/
CMH), and Hoeffding's D.
Emitted only when the corresponding module's `bench_*` self-check
completes on its SYNTHETIC fixture.
"""

from __future__ import annotations

import math

_MARDIA_SEED = 20261231 + 426
_MANTEL_SEED = 20261231 + 427
_MORAN_SEED = 20261231 + 428
_FRIEDMAN_SEED = 20261231 + 429
_CONTINGENCY_SEED = 20261231 + 430
_HOEFFDING_SEED = 20261231 + 431

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


def bench_mardia() -> dict[str, float]:
    try:
        from quant_fund.models.mardia import (
            bench_mardia as _core,
        )
    except ImportError:  # pragma: no cover - optional
        return {}
    try:
        out = _core(seed=_MARDIA_SEED)
    except (ValueError, RuntimeError, FloatingPointError, KeyError, TypeError):
        return {}
    return _isinstance_floats(out)


def bench_mantel() -> dict[str, float]:
    try:
        from quant_fund.models.mantel import (
            bench_mantel as _core,
        )
    except ImportError:  # pragma: no cover - optional
        return {}
    try:
        out = _core(seed=_MANTEL_SEED)
    except (ValueError, RuntimeError, FloatingPointError, KeyError, TypeError):
        return {}
    return _isinstance_floats(out)


def bench_moran() -> dict[str, float]:
    try:
        from quant_fund.models.moran import (
            bench_moran as _core,
        )
    except ImportError:  # pragma: no cover - optional
        return {}
    try:
        out = _core(seed=_MORAN_SEED)
    except (ValueError, RuntimeError, FloatingPointError, KeyError, TypeError):
        return {}
    return _isinstance_floats(out)


def bench_friedman() -> dict[str, float]:
    try:
        from quant_fund.models.friedman import bench_friedman as _core
    except ImportError:  # pragma: no cover - optional
        return {}
    try:
        out = _core(seed=_FRIEDMAN_SEED)
    except (ValueError, RuntimeError, FloatingPointError, KeyError, TypeError):
        return {}
    return _isinstance_floats(out)


def bench_contingency() -> dict[str, float]:
    try:
        from quant_fund.models.contingency import (
            bench_contingency as _core,
        )
    except ImportError:  # pragma: no cover - optional
        return {}
    try:
        out = _core(seed=_CONTINGENCY_SEED)
    except (ValueError, RuntimeError, FloatingPointError, KeyError, TypeError):
        return {}
    return _isinstance_floats(out)


def bench_hoeffding() -> dict[str, float]:
    try:
        from quant_fund.models.hoeffding import (
            bench_hoeffding as _core,
        )
    except ImportError:  # pragma: no cover - optional
        return {}
    try:
        out = _core(seed=_HOEFFDING_SEED)
    except (ValueError, RuntimeError, FloatingPointError, KeyError, TypeError):
        return {}
    return _isinstance_floats(out)
