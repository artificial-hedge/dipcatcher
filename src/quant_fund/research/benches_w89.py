"""Wave-89 optional scorecard families — the classical
hypothesis-testing canon: Kolmogorov-Smirnov /
Cramer-von Mises / Anderson-Darling (1952/1954)
empirical-distribution-function tests, Shapiro-Wilk
(1965)/Jarque-Bera (1980)/D'Agostino-Pearson (1973)
normality, Levene (1960)/Brown-Forsythe (1974)/
Fligner-Killeen (1976)/O'Brien (1979) scale
homogeneity, Ansari-Bradley (1960)/Mood (1954)/
Klotz (1962)/Conover (1980)/Gastwirth (1965) rank
scale tests, Goldfeld-Quandt (1965)/Park (1966)/
Glejser (1969)/Breusch-Pagan (1979)/White (1980)
heteroskedasticity regressions, and Durbin-Watson
(1950)/Durbin-h (1970)/Breusch-Godfrey (1978)/
Ljung-Box (1978) serial-correlation diagnostics.
Emitted only when the corresponding module's
`bench_*` self-check completes on its SYNTHETIC
fixture.
"""

from __future__ import annotations

import math

_EDF_SEED = 20261231 + 522
_NORM_SEED = 20261231 + 523
_SCALE_SEED = 20261231 + 524
_SCORE_SEED = 20261231 + 525
_HET_SEED = 20261231 + 526
_SERIAL_SEED = 20261231 + 527

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


def bench_edf_tests() -> dict[str, float]:
    try:
        from quant_fund.models.edf_tests import bench_edf as _core
    except ImportError:  # pragma: no cover - optional
        return {}
    try:
        out = _core(seed=_EDF_SEED)
    except (ValueError, RuntimeError, FloatingPointError, KeyError, TypeError):
        return {}
    return _isinstance_floats(out)


def bench_normality_tests() -> dict[str, float]:
    try:
        from quant_fund.models.normality_tests import (
            bench_normality as _core,
        )
    except ImportError:  # pragma: no cover - optional
        return {}
    try:
        out = _core(seed=_NORM_SEED)
    except (ValueError, RuntimeError, FloatingPointError, KeyError, TypeError):
        return {}
    return _isinstance_floats(out)


def bench_scale_homogeneity() -> dict[str, float]:
    try:
        from quant_fund.models.scale_homogeneity import (
            bench_scale as _core,
        )
    except ImportError:  # pragma: no cover - optional
        return {}
    try:
        out = _core(seed=_SCALE_SEED)
    except (ValueError, RuntimeError, FloatingPointError, KeyError, TypeError):
        return {}
    return _isinstance_floats(out)


def bench_score_scale() -> dict[str, float]:
    try:
        from quant_fund.models.score_scale import (
            bench_score_scale as _core,
        )
    except ImportError:  # pragma: no cover - optional
        return {}
    try:
        out = _core(seed=_SCORE_SEED)
    except (ValueError, RuntimeError, FloatingPointError, KeyError, TypeError):
        return {}
    return _isinstance_floats(out)


def bench_het_regressions() -> dict[str, float]:
    try:
        from quant_fund.models.het_regressions import bench_het as _core
    except ImportError:  # pragma: no cover - optional
        return {}
    try:
        out = _core(seed=_HET_SEED)
    except (ValueError, RuntimeError, FloatingPointError, KeyError, TypeError):
        return {}
    return _isinstance_floats(out)


def bench_serial_diagnostics() -> dict[str, float]:
    try:
        from quant_fund.models.serial_diagnostics import (
            bench_serial as _core,
        )
    except ImportError:  # pragma: no cover - optional
        return {}
    try:
        out = _core(seed=_SERIAL_SEED)
    except (ValueError, RuntimeError, FloatingPointError, KeyError, TypeError):
        return {}
    return _isinstance_floats(out)
