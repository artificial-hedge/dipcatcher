"""Wave-70 optional scorecard families — cyclostationary signal
analysis, adaptive filterbanks, rank-based inference, latent
correlation, compositional geometry, and high-dimensional screening.
Emitted only when the corresponding module's `bench_*` self-check
completes on its SYNTHETIC fixture.
"""

from __future__ import annotations

import math

_CYCLOSTATIONARY_SEED = 20261231 + 408
_EMPIRICAL_WAVELETS_SEED = 20261231 + 409
_NONPARAMETRIC_TESTS_SEED = 20261231 + 410
_POLYCHORIC_SEED = 20261231 + 411
_COMPOSITIONAL_SEED = 20261231 + 412
_SURE_SCREENING_SEED = 20261231 + 413

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


def bench_cyclostationary() -> dict[str, float]:
    try:
        from quant_fund.models.cyclostationary import (
            bench_cyclostationary as _core,
        )
    except ImportError:  # pragma: no cover - optional
        return {}
    try:
        out = _core(seed=_CYCLOSTATIONARY_SEED)
    except (ValueError, RuntimeError, FloatingPointError, KeyError, TypeError):
        return {}
    return _isinstance_floats(out)


def bench_empirical_wavelets() -> dict[str, float]:
    try:
        from quant_fund.models.empirical_wavelets import (
            bench_empirical_wavelets as _core,
        )
    except ImportError:  # pragma: no cover - optional
        return {}
    try:
        out = _core(seed=_EMPIRICAL_WAVELETS_SEED)
    except (ValueError, RuntimeError, FloatingPointError, KeyError, TypeError):
        return {}
    return _isinstance_floats(out)


def bench_nonparametric_tests() -> dict[str, float]:
    try:
        from quant_fund.models.nonparametric_tests import (
            bench_nonparametric_tests as _core,
        )
    except ImportError:  # pragma: no cover - optional
        return {}
    try:
        out = _core(seed=_NONPARAMETRIC_TESTS_SEED)
    except (ValueError, RuntimeError, FloatingPointError, KeyError, TypeError):
        return {}
    return _isinstance_floats(out)


def bench_polychoric() -> dict[str, float]:
    try:
        from quant_fund.models.polychoric import bench_polychoric as _core
    except ImportError:  # pragma: no cover - optional
        return {}
    try:
        out = _core(seed=_POLYCHORIC_SEED)
    except (ValueError, RuntimeError, FloatingPointError, KeyError, TypeError):
        return {}
    return _isinstance_floats(out)


def bench_compositional() -> dict[str, float]:
    try:
        from quant_fund.models.compositional import (
            bench_compositional as _core,
        )
    except ImportError:  # pragma: no cover - optional
        return {}
    try:
        out = _core(seed=_COMPOSITIONAL_SEED)
    except (ValueError, RuntimeError, FloatingPointError, KeyError, TypeError):
        return {}
    return _isinstance_floats(out)


def bench_sure_screening() -> dict[str, float]:
    try:
        from quant_fund.models.sure_screening import (
            bench_sure_screening as _core,
        )
    except ImportError:  # pragma: no cover - optional
        return {}
    try:
        out = _core(seed=_SURE_SCREENING_SEED)
    except (ValueError, RuntimeError, FloatingPointError, KeyError, TypeError):
        return {}
    return _isinstance_floats(out)
