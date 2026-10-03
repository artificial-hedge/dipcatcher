"""Wave-61 adapter benches — optional scorecard families.

Each adapter calls a SYNTHETIC bench on the wave-61 models and
returns a small metric dict consumed by the scorecard runner.
Adapters are honest by contract: they propagate ``score`` and
label every metric ``synthetic_*``. Any import/shape/exception
failure returns ``{}`` so the family is skipped, never faked.

Families (seeds 20261231 + 354..359):
- ``log_concave``      — Dümbgen-Rufibach log-concave density MLE
- ``dtw_warp``         — Sakoe-Chiba banded DTW + warp registration
- ``chow_lin``         — Chow-Lin GLS + Denton temporal disaggregation
- ``chen_tiao_outliers`` — Chen-Liu-Tiao AO/IO/LS/TC battery
- ``beta_ar``          — Rocha-Cribari-Neto beta autoregression
- ``ingarch``          — Ferland-Latour-Oraichi Poisson INGARCH
"""

from __future__ import annotations

import math

_SEED = 20261231
_LOG_CONCAVE_SEED = _SEED + 354
_DTW_WARP_SEED = _SEED + 355
_CHOW_LIN_SEED = _SEED + 356
_CHEN_TIAO_SEED = _SEED + 357
_BETA_AR_SEED = _SEED + 358
_INGARCH_SEED = _SEED + 359

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


def bench_log_concave() -> dict[str, float]:
    try:
        from quant_fund.models.log_concave import bench_log_concave as _core
    except ImportError:  # pragma: no cover - optional
        return {}
    try:
        out = _core(seed=_LOG_CONCAVE_SEED)
    except (ValueError, RuntimeError, FloatingPointError, KeyError, TypeError):
        return {}
    return _isinstance_floats(out)


def bench_dtw_warp() -> dict[str, float]:
    try:
        from quant_fund.models.dtw_warp import bench_dtw as _core
    except ImportError:  # pragma: no cover - optional
        return {}
    try:
        out = _core(seed=_DTW_WARP_SEED)
    except (ValueError, RuntimeError, FloatingPointError, KeyError, TypeError):
        return {}
    return _isinstance_floats(out)


def bench_chow_lin() -> dict[str, float]:
    try:
        from quant_fund.models.chow_lin import bench_chow_lin as _core
    except ImportError:  # pragma: no cover - optional
        return {}
    try:
        out = _core(seed=_CHOW_LIN_SEED)
    except (ValueError, RuntimeError, FloatingPointError, KeyError, TypeError):
        return {}
    return _isinstance_floats(out)


def bench_chen_tiao_outliers() -> dict[str, float]:
    try:
        from quant_fund.models.chen_tiao_outliers import (
            bench_outliers as _core,
        )
    except ImportError:  # pragma: no cover - optional
        return {}
    try:
        out = _core(seed=_CHEN_TIAO_SEED)
    except (ValueError, RuntimeError, FloatingPointError, KeyError, TypeError):
        return {}
    return _isinstance_floats(out)


def bench_beta_ar() -> dict[str, float]:
    try:
        from quant_fund.models.beta_ar import bench_beta_ar as _core
    except ImportError:  # pragma: no cover - optional
        return {}
    try:
        out = _core(seed=_BETA_AR_SEED)
    except (ValueError, RuntimeError, FloatingPointError, KeyError, TypeError):
        return {}
    return _isinstance_floats(out)


def bench_ingarch() -> dict[str, float]:
    try:
        from quant_fund.models.ingarch import bench_ingarch as _core
    except ImportError:  # pragma: no cover - optional
        return {}
    try:
        out = _core(seed=_INGARCH_SEED)
    except (ValueError, RuntimeError, FloatingPointError, KeyError, TypeError):
        return {}
    return _isinstance_floats(out)
