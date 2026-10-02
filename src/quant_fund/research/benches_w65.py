"""Wave-65 adapter benches — optional scorecard families.

Each adapter calls a SYNTHETIC bench on the wave-65 models and
returns a small metric dict consumed by the scorecard runner.
Adapters are honest by contract: they propagate ``score`` and
label every metric ``synthetic_*``. Any import/shape/exception
failure returns ``{}`` so the family is skipped, never faked.

Families (seeds 20261231 + 378..383):
- ``libor_market``     — LFM terminal-measure forwards, caplets, swaptions
- ``cos_method``       — Fang-Oosterlee Fourier-cosine option pricing
- ``obizhaeva_wang``   — transient-impact optimal execution schedule
- ``spread_options``   — Margrabe/Kirk exchange-spread + quanto options
- ``esscher``          — Gerber-Shiu Esscher-measure option pricing
- ``shadow_rate``      — Wu-Xia shadow-rate EKF term structure
"""

from __future__ import annotations

import math

_SEED = 20261231
_LIBOR_MARKET_SEED = _SEED + 378
_COS_METHOD_SEED = _SEED + 379
_OBIZHAEVA_WANG_SEED = _SEED + 380
_SPREAD_OPTIONS_SEED = _SEED + 381
_ESSCHER_SEED = _SEED + 382
_SHADOW_RATE_SEED = _SEED + 383

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


def bench_libor_market() -> dict[str, float]:
    try:
        from quant_fund.models.libor_market import bench_libor_market as _core
    except ImportError:  # pragma: no cover - optional
        return {}
    try:
        out = _core(seed=_LIBOR_MARKET_SEED)
    except (ValueError, RuntimeError, FloatingPointError, KeyError, TypeError):
        return {}
    return _isinstance_floats(out)


def bench_cos_method() -> dict[str, float]:
    try:
        from quant_fund.models.cos_method import bench_cos_method as _core
    except ImportError:  # pragma: no cover - optional
        return {}
    try:
        out = _core(seed=_COS_METHOD_SEED)
    except (ValueError, RuntimeError, FloatingPointError, KeyError, TypeError):
        return {}
    return _isinstance_floats(out)


def bench_obizhaeva_wang() -> dict[str, float]:
    try:
        from quant_fund.models.obizhaeva_wang import bench_obizhaeva_wang as _core
    except ImportError:  # pragma: no cover - optional
        return {}
    try:
        out = _core(seed=_OBIZHAEVA_WANG_SEED)
    except (ValueError, RuntimeError, FloatingPointError, KeyError, TypeError):
        return {}
    return _isinstance_floats(out)


def bench_spread_options() -> dict[str, float]:
    try:
        from quant_fund.models.spread_options import bench_spread_options as _core
    except ImportError:  # pragma: no cover - optional
        return {}
    try:
        out = _core(seed=_SPREAD_OPTIONS_SEED)
    except (ValueError, RuntimeError, FloatingPointError, KeyError, TypeError):
        return {}
    return _isinstance_floats(out)


def bench_esscher() -> dict[str, float]:
    try:
        from quant_fund.models.esscher import bench_esscher as _core
    except ImportError:  # pragma: no cover - optional
        return {}
    try:
        out = _core(seed=_ESSCHER_SEED)
    except (ValueError, RuntimeError, FloatingPointError, KeyError, TypeError):
        return {}
    return _isinstance_floats(out)


def bench_shadow_rate() -> dict[str, float]:
    try:
        from quant_fund.models.shadow_rate import bench_shadow_rate as _core
    except ImportError:  # pragma: no cover - optional
        return {}
    try:
        out = _core(seed=_SHADOW_RATE_SEED)
    except (ValueError, RuntimeError, FloatingPointError, KeyError, TypeError):
        return {}
    return _isinstance_floats(out)
