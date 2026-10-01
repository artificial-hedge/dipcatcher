"""Wave-54 optional benchmark adapters.

Each adapter runs one wave-54 module's self-check and returns a
``{metric_key: float}`` blob — all ``synthetic_*`` proper diagnostics, never
market evidence. A lane that is unavailable (ImportError) or whose numbers are
non-finite — the ``_finite_blob`` gate, waves 15-51 precedent — degrades to
``{}`` so the scorecard omits it cleanly.
"""

from __future__ import annotations

import math

_SEED = 20261231  # wave-54 stamp seed

_PESARAN_TIMMERMANN_SEED = _SEED + 311
_GIACOMINI_ROSSI_SEED = _SEED + 312
_MULLER_WATSON_SEED = _SEED + 313
_ROMANO_WOLF_SEED = _SEED + 314
_CDR_SEED = _SEED + 315
_DANIELSSON_DEVRIES_SEED = _SEED + 316

_FORBIDDEN_TOKENS = frozenset({"sharpe", "sortino", "calmar", "pnl", "nav"})


def _finite_blob(mapped: dict[str, float]) -> dict[str, float]:
    """``{}`` unless every emitted value is finite (ruff-bench contract)."""
    if all(math.isfinite(v) for v in mapped.values()):
        return mapped
    return {}


def _isinstance_floats(
    raw: dict[str, float] | dict[str, object],
) -> dict[str, float]:
    """Float-coerce a lane blob, dropping str stamps, runtime telemetry,
    and any key carrying a forbidden headline metric token."""
    return {
        k: float(v)
        for k, v in raw.items()
        if isinstance(v, (int, float)) and _FORBIDDEN_TOKENS.isdisjoint(k.lower().split("_"))
    }


def bench_pesaran_timmermann() -> dict[str, float]:
    try:
        from quant_fund.models.pesaran_timmermann import (
            bench_pesaran_timmermann as _core,
        )
    except ImportError:
        return {}
    try:
        return _finite_blob(_isinstance_floats(_core(seed=_PESARAN_TIMMERMANN_SEED)))
    except (ValueError, RuntimeError, FloatingPointError, KeyError, TypeError):
        return {}


def bench_giacomini_rossi() -> dict[str, float]:
    try:
        from quant_fund.models.giacomini_rossi import (
            bench_giacomini_rossi as _core,
        )
    except ImportError:
        return {}
    try:
        return _finite_blob(_isinstance_floats(_core(seed=_GIACOMINI_ROSSI_SEED)))
    except (ValueError, RuntimeError, FloatingPointError, KeyError, TypeError):
        return {}


def bench_muller_watson() -> dict[str, float]:
    try:
        from quant_fund.models.muller_watson import (
            bench_muller_watson as _core,
        )
    except ImportError:
        return {}
    try:
        return _finite_blob(_isinstance_floats(_core(seed=_MULLER_WATSON_SEED)))
    except (ValueError, RuntimeError, FloatingPointError, KeyError, TypeError):
        return {}


def bench_romano_wolf() -> dict[str, float]:
    try:
        from quant_fund.models.romano_wolf import (
            bench_romano_wolf as _core,
        )
    except ImportError:
        return {}
    try:
        return _finite_blob(_isinstance_floats(_core(seed=_ROMANO_WOLF_SEED)))
    except (ValueError, RuntimeError, FloatingPointError, KeyError, TypeError):
        return {}


def bench_christensen_diebold_rudebusch() -> dict[str, float]:
    try:
        from quant_fund.models.christensen_diebold_rudebusch import (
            bench_christensen_diebold_rudebusch as _core,
        )
    except ImportError:
        return {}
    try:
        return _finite_blob(_isinstance_floats(_core(seed=_CDR_SEED)))
    except (ValueError, RuntimeError, FloatingPointError, KeyError, TypeError):
        return {}


def bench_danielsson_devries() -> dict[str, float]:
    try:
        from quant_fund.models.danielsson_devries import (
            bench_danielsson_devries as _core,
        )
    except ImportError:
        return {}
    try:
        return _finite_blob(_isinstance_floats(_core(seed=_DANIELSSON_DEVRIES_SEED)))
    except (ValueError, RuntimeError, FloatingPointError, KeyError, TypeError):
        return {}
