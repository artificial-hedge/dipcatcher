"""Wave-60 adapter benches — optional scorecard families.

Each adapter calls a SYNTHETIC bench on the wave-60 models and
returns a small metric dict consumed by the scorecard runner.
Adapters are honest by contract: they propagate ``score`` and
label every metric ``synthetic_*``. Any import/shape/exception
failure returns ``{}`` so the family is skipped, never faked.

Families (seeds 20261231 + 348..353):
- ``kalman_em``       — Shumway-Stoffer EM state-space estimation
- ``fractional_coint`` — GPH residual + Marinucci-Robinson
                          fractional cointegration
- ``star_model``      — Terasvirta LSTAR/ESTAR + LST LM3 test
- ``garch_in_mean``   — Engle-Lilien-Robins GARCH-M
- ``log_acd``         — Bauwens-Giot log-ACD (Weibull/lognormal)
- ``wigner_ville``    — Wigner-Ville / pseudo-WVD time-frequency
"""

from __future__ import annotations

import math

_SEED = 20261231
_KALMAN_EM_SEED = _SEED + 348
_FRACTIONAL_COINT_SEED = _SEED + 349
_STAR_MODEL_SEED = _SEED + 350
_GARCH_IN_MEAN_SEED = _SEED + 351
_LOG_ACD_SEED = _SEED + 352
_WIGNER_VILLE_SEED = _SEED + 353

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


def bench_kalman_em() -> dict[str, float]:
    try:
        from quant_fund.models.kalman_em import bench_kalman_em as _core
    except ImportError:  # pragma: no cover - optional
        return {}
    try:
        out = _core(seed=_KALMAN_EM_SEED)
    except (ValueError, RuntimeError, FloatingPointError, KeyError, TypeError):
        return {}
    return _isinstance_floats(out)


def bench_fractional_coint() -> dict[str, float]:
    try:
        from quant_fund.models.fractional_coint import (
            bench_fractional_coint as _core,
        )
    except ImportError:  # pragma: no cover - optional
        return {}
    try:
        out = _core(seed=_FRACTIONAL_COINT_SEED)
    except (ValueError, RuntimeError, FloatingPointError, KeyError, TypeError):
        return {}
    return _isinstance_floats(out)


def bench_star_model() -> dict[str, float]:
    try:
        from quant_fund.models.star_model import bench_star as _core
    except ImportError:  # pragma: no cover - optional
        return {}
    try:
        out = _core(seed=_STAR_MODEL_SEED)
    except (ValueError, RuntimeError, FloatingPointError, KeyError, TypeError):
        return {}
    return _isinstance_floats(out)


def bench_garch_in_mean() -> dict[str, float]:
    try:
        from quant_fund.models.garch_in_mean import (
            bench_garch_in_mean as _core,
        )
    except ImportError:  # pragma: no cover - optional
        return {}
    try:
        out = _core(seed=_GARCH_IN_MEAN_SEED)
    except (ValueError, RuntimeError, FloatingPointError, KeyError, TypeError):
        return {}
    return _isinstance_floats(out)


def bench_log_acd() -> dict[str, float]:
    try:
        from quant_fund.models.log_acd import bench_log_acd as _core
    except ImportError:  # pragma: no cover - optional
        return {}
    try:
        out = _core(seed=_LOG_ACD_SEED)
    except (ValueError, RuntimeError, FloatingPointError, KeyError, TypeError):
        return {}
    return _isinstance_floats(out)


def bench_wigner_ville() -> dict[str, float]:
    try:
        from quant_fund.models.wigner_ville import (
            bench_wigner_ville as _core,
        )
    except ImportError:  # pragma: no cover - optional
        return {}
    try:
        out = _core(seed=_WIGNER_VILLE_SEED)
    except (ValueError, RuntimeError, FloatingPointError, KeyError, TypeError):
        return {}
    return _isinstance_floats(out)
