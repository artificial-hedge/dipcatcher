"""Wave-76 optional scorecard families — Horvitz-
Thompson / Hajek design-based survey estimation,
post-stratification + iterative raking, GREG
calibration, Fay-Herriot EBLUP small-area
estimation, cluster sampling, and Kish design
effects.
Emitted only when the corresponding module's `bench_*` self-check
completes on its SYNTHETIC fixture.
"""

from __future__ import annotations

import math

_HORVITZ_THOMPSON_SEED = 20261231 + 444
_POSTSTRAT_SEED = 20261231 + 445
_CALIBRATION_SURVEY_SEED = 20261231 + 446
_FAY_HERRIOT_SEED = 20261231 + 447
_CLUSTER_SAMPLING_SEED = 20261231 + 448
_DESIGN_EFFECTS_SEED = 20261231 + 449

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


def bench_horvitz_thompson() -> dict[str, float]:
    try:
        from quant_fund.models.horvitz_thompson import (
            bench_horvitz_thompson as _core,
        )
    except ImportError:  # pragma: no cover - optional
        return {}
    try:
        out = _core(seed=_HORVITZ_THOMPSON_SEED)
    except (ValueError, RuntimeError, FloatingPointError, KeyError, TypeError):
        return {}
    return _isinstance_floats(out)


def bench_poststrat() -> dict[str, float]:
    try:
        from quant_fund.models.poststrat import (
            bench_poststrat as _core,
        )
    except ImportError:  # pragma: no cover - optional
        return {}
    try:
        out = _core(seed=_POSTSTRAT_SEED)
    except (ValueError, RuntimeError, FloatingPointError, KeyError, TypeError):
        return {}
    return _isinstance_floats(out)


def bench_calibration_survey() -> dict[str, float]:
    try:
        from quant_fund.models.calibration_survey import (
            bench_calibration_survey as _core,
        )
    except ImportError:  # pragma: no cover - optional
        return {}
    try:
        out = _core(seed=_CALIBRATION_SURVEY_SEED)
    except (ValueError, RuntimeError, FloatingPointError, KeyError, TypeError):
        return {}
    return _isinstance_floats(out)


def bench_fay_herriot() -> dict[str, float]:
    try:
        from quant_fund.models.fay_herriot import bench_fay_herriot as _core
    except ImportError:  # pragma: no cover - optional
        return {}
    try:
        out = _core(seed=_FAY_HERRIOT_SEED)
    except (ValueError, RuntimeError, FloatingPointError, KeyError, TypeError):
        return {}
    return _isinstance_floats(out)


def bench_cluster_sampling() -> dict[str, float]:
    try:
        from quant_fund.models.cluster_sampling import (
            bench_cluster_sampling as _core,
        )
    except ImportError:  # pragma: no cover - optional
        return {}
    try:
        out = _core(seed=_CLUSTER_SAMPLING_SEED)
    except (ValueError, RuntimeError, FloatingPointError, KeyError, TypeError):
        return {}
    return _isinstance_floats(out)


def bench_design_effects() -> dict[str, float]:
    try:
        from quant_fund.models.design_effects import (
            bench_design_effects as _core,
        )
    except ImportError:  # pragma: no cover - optional
        return {}
    try:
        out = _core(seed=_DESIGN_EFFECTS_SEED)
    except (ValueError, RuntimeError, FloatingPointError, KeyError, TypeError):
        return {}
    return _isinstance_floats(out)
