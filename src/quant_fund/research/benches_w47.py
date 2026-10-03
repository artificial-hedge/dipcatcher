"""Wave-46 optional benchmark adapters.

Each adapter runs one wave-47 module's self-check and returns a
``{metric_key: float}`` blob — all ``synthetic_*`` proper diagnostics, never
market evidence. A lane that is unavailable (ImportError) or whose numbers are
non-finite — the ``_finite_blob`` gate, waves 15-46 precedent — degrades to
``{}`` so the scorecard omits it cleanly.
"""

from __future__ import annotations

import math

_SEED = 20261231  # wave-47 stamp seed

_ROSENBAUM_SENSITIVITY_SEED = _SEED + 269
_AIPW_ATE_SEED = _SEED + 270
_CAVI_GMM_SEED = _SEED + 271
_PESARAN_CD_SEED = _SEED + 272
_HAUSMAN_TESTS_SEED = _SEED + 273
_CUSUM_MONITOR_SEED = _SEED + 274

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


def bench_rosenbaum_sensitivity() -> dict[str, float]:
    try:
        from quant_fund.models.rosenbaum_sensitivity import (
            bench_rosenbaum_sensitivity as _core,
        )
    except ImportError:
        return {}
    try:
        return _finite_blob(_isinstance_floats(_core(seed=_ROSENBAUM_SENSITIVITY_SEED)))
    except (ValueError, RuntimeError, FloatingPointError, KeyError, TypeError):
        return {}


def bench_aipw_ate() -> dict[str, float]:
    try:
        from quant_fund.models.aipw_ate import (
            bench_aipw_ate as _core,
        )
    except ImportError:
        return {}
    try:
        return _finite_blob(_isinstance_floats(_core(seed=_AIPW_ATE_SEED)))
    except (ValueError, RuntimeError, FloatingPointError, KeyError, TypeError):
        return {}


def bench_cavi_gmm() -> dict[str, float]:
    try:
        from quant_fund.models.cavi_gmm import (
            bench_cavi_gmm as _core,
        )
    except ImportError:
        return {}
    try:
        return _finite_blob(_isinstance_floats(_core(seed=_CAVI_GMM_SEED)))
    except (ValueError, RuntimeError, FloatingPointError, KeyError, TypeError):
        return {}


def bench_pesaran_cd() -> dict[str, float]:
    try:
        from quant_fund.models.pesaran_cd import (
            bench_pesaran_cd as _core,
        )
    except ImportError:
        return {}
    try:
        return _finite_blob(_isinstance_floats(_core(seed=_PESARAN_CD_SEED)))
    except (ValueError, RuntimeError, FloatingPointError, KeyError, TypeError):
        return {}


def bench_hausman_tests() -> dict[str, float]:
    try:
        from quant_fund.models.hausman_tests import (
            bench_hausman_tests as _core,
        )
    except ImportError:
        return {}
    try:
        return _finite_blob(_isinstance_floats(_core(seed=_HAUSMAN_TESTS_SEED)))
    except (ValueError, RuntimeError, FloatingPointError, KeyError, TypeError):
        return {}


def bench_cusum_monitor() -> dict[str, float]:
    try:
        from quant_fund.models.cusum_monitor import (
            bench_cusum_monitor as _core,
        )
    except ImportError:
        return {}
    try:
        return _finite_blob(_isinstance_floats(_core(seed=_CUSUM_MONITOR_SEED)))
    except (ValueError, RuntimeError, FloatingPointError, KeyError, TypeError):
        return {}
