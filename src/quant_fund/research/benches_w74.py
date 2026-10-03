"""Wave-74 optional scorecard families — rank-based
dispersion (Siegel-Tukey, Ansari-Bradley), Mood's
median, Cochran's Q, Quade weighted block ranks,
van der Waerden normal scores, and Dunn post-hoc.
Emitted only when the corresponding module's `bench_*` self-check
completes on its SYNTHETIC fixture.
"""

from __future__ import annotations

import math

_DISPERSION_TESTS_SEED = 20261231 + 432
_MEDIAN_TESTS_SEED = 20261231 + 433
_COCHRAN_Q_SEED = 20261231 + 434
_QUADE_SEED = 20261231 + 435
_VAN_DER_WAERDEN_SEED = 20261231 + 436
_DUNN_TEST_SEED = 20261231 + 437

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


def bench_dispersion_tests() -> dict[str, float]:
    try:
        from quant_fund.models.dispersion_tests import (
            bench_dispersion_tests as _core,
        )
    except ImportError:  # pragma: no cover - optional
        return {}
    try:
        out = _core(seed=_DISPERSION_TESTS_SEED)
    except (ValueError, RuntimeError, FloatingPointError, KeyError, TypeError):
        return {}
    return _isinstance_floats(out)


def bench_median_tests() -> dict[str, float]:
    try:
        from quant_fund.models.median_tests import (
            bench_median_tests as _core,
        )
    except ImportError:  # pragma: no cover - optional
        return {}
    try:
        out = _core(seed=_MEDIAN_TESTS_SEED)
    except (ValueError, RuntimeError, FloatingPointError, KeyError, TypeError):
        return {}
    return _isinstance_floats(out)


def bench_cochran_q() -> dict[str, float]:
    try:
        from quant_fund.models.cochran_q import (
            bench_cochran_q as _core,
        )
    except ImportError:  # pragma: no cover - optional
        return {}
    try:
        out = _core(seed=_COCHRAN_Q_SEED)
    except (ValueError, RuntimeError, FloatingPointError, KeyError, TypeError):
        return {}
    return _isinstance_floats(out)


def bench_quade() -> dict[str, float]:
    try:
        from quant_fund.models.quade import bench_quade as _core
    except ImportError:  # pragma: no cover - optional
        return {}
    try:
        out = _core(seed=_QUADE_SEED)
    except (ValueError, RuntimeError, FloatingPointError, KeyError, TypeError):
        return {}
    return _isinstance_floats(out)


def bench_van_der_waerden() -> dict[str, float]:
    try:
        from quant_fund.models.van_der_waerden import (
            bench_van_der_waerden as _core,
        )
    except ImportError:  # pragma: no cover - optional
        return {}
    try:
        out = _core(seed=_VAN_DER_WAERDEN_SEED)
    except (ValueError, RuntimeError, FloatingPointError, KeyError, TypeError):
        return {}
    return _isinstance_floats(out)


def bench_dunn_test() -> dict[str, float]:
    try:
        from quant_fund.models.dunn_test import (
            bench_dunn_test as _core,
        )
    except ImportError:  # pragma: no cover - optional
        return {}
    try:
        out = _core(seed=_DUNN_TEST_SEED)
    except (ValueError, RuntimeError, FloatingPointError, KeyError, TypeError):
        return {}
    return _isinstance_floats(out)
