"""Wave-82 optional scorecard families — DeLong
(1988) correlated-AUC variance estimation and
paired comparison, Passing-Bablok (1983) robust
method-comparison regression + Deming orthogonal
fit, McNemar/Bowker/Stuart-Maxwell/Bhapkar marginal
homogeneity, Mardia-Watson-Wheeler/Rao-spacing/
Watson-Beran circular tests, Samejima graded-response
+ partial-credit IRT joint ML, and Welch (1951)
heteroscedastic ANOVA with Games-Howell post-hoc.
Emitted only when the corresponding module's `bench_*` self-check
completes on its SYNTHETIC fixture.
"""

from __future__ import annotations

import math

_DELONG_AUC_SEED = 20261231 + 480
_PASSING_BABLOK_SEED = 20261231 + 481
_MARGINAL_HOMOGENEITY_SEED = 20261231 + 482
_CIRCULAR_TESTS_SEED = 20261231 + 483
_GRADED_IRT_SEED = 20261231 + 484
_WELCH_ANOVA_SEED = 20261231 + 485

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


def bench_delong_auc() -> dict[str, float]:
    try:
        from quant_fund.models.delong_auc import bench_delong_auc as _core
    except ImportError:  # pragma: no cover - optional
        return {}
    try:
        out = _core(seed=_DELONG_AUC_SEED)
    except (ValueError, RuntimeError, FloatingPointError, KeyError, TypeError):
        return {}
    return _isinstance_floats(out)


def bench_passing_bablok() -> dict[str, float]:
    try:
        from quant_fund.models.passing_bablok import (
            bench_passing_bablok as _core,
        )
    except ImportError:  # pragma: no cover - optional
        return {}
    try:
        out = _core(seed=_PASSING_BABLOK_SEED)
    except (ValueError, RuntimeError, FloatingPointError, KeyError, TypeError):
        return {}
    return _isinstance_floats(out)


def bench_marginal_homogeneity() -> dict[str, float]:
    try:
        from quant_fund.models.marginal_homogeneity import (
            bench_marginal_homogeneity as _core,
        )
    except ImportError:  # pragma: no cover - optional
        return {}
    try:
        out = _core(seed=_MARGINAL_HOMOGENEITY_SEED)
    except (ValueError, RuntimeError, FloatingPointError, KeyError, TypeError):
        return {}
    return _isinstance_floats(out)


def bench_circular_tests() -> dict[str, float]:
    try:
        from quant_fund.models.circular_tests import (
            bench_circular_tests as _core,
        )
    except ImportError:  # pragma: no cover - optional
        return {}
    try:
        out = _core(seed=_CIRCULAR_TESTS_SEED)
    except (ValueError, RuntimeError, FloatingPointError, KeyError, TypeError):
        return {}
    return _isinstance_floats(out)


def bench_graded_irt() -> dict[str, float]:
    try:
        from quant_fund.models.graded_irt import bench_graded_irt as _core
    except ImportError:  # pragma: no cover - optional
        return {}
    try:
        out = _core(seed=_GRADED_IRT_SEED)
    except (ValueError, RuntimeError, FloatingPointError, KeyError, TypeError):
        return {}
    return _isinstance_floats(out)


def bench_welch_anova() -> dict[str, float]:
    try:
        from quant_fund.models.welch_anova import bench_welch_anova as _core
    except ImportError:  # pragma: no cover - optional
        return {}
    try:
        out = _core(seed=_WELCH_ANOVA_SEED)
    except (ValueError, RuntimeError, FloatingPointError, KeyError, TypeError):
        return {}
    return _isinstance_floats(out)
