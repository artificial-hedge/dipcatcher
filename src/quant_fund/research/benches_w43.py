"""Wave-43 optional benchmark adapters.

Each adapter runs one wave-43 module's self-check and returns a
``{metric_key: float}`` blob — all ``synthetic_*`` proper diagnostics, never
market evidence. A lane that is unavailable (ImportError) or whose numbers are
non-finite — the ``_finite_blob`` gate, waves 15-42 precedent — degrades to
``{}`` so the scorecard omits it cleanly.
"""

from __future__ import annotations

import math

_SEED = 20261231  # wave-43 stamp seed

# --- per-family seeds -------------------------------------------------------
_BAI_PERRON_SEED = _SEED + 245
_FAVAR_SEED = _SEED + 246
_PANEL_COINT_SEED = _SEED + 247
_VUONG_TEST_SEED = _SEED + 248
_MERTON_MODEL_SEED = _SEED + 249
_WHITE_REALITY_SEED = _SEED + 250


def _finite_blob(mapped: dict[str, float]) -> dict[str, float]:
    """``{}`` unless every emitted value is finite (ruff-bench contract)."""
    if all(math.isfinite(v) for v in mapped.values()):
        return mapped
    return {}


_FORBIDDEN_TOKENS = frozenset({"sharpe", "sortino", "calmar", "pnl", "nav"})


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


def bench_bai_perron() -> dict[str, float]:
    try:
        from quant_fund.models.bai_perron import (
            bench_bai_perron as _core,
        )
    except ImportError:
        return {}
    try:
        return _finite_blob(_isinstance_floats(_core(seed=_BAI_PERRON_SEED)))
    except (ValueError, RuntimeError, FloatingPointError, KeyError, TypeError):
        return {}


def bench_favar() -> dict[str, float]:
    try:
        from quant_fund.models.favar import (
            bench_favar as _core,
        )
    except ImportError:
        return {}
    try:
        return _finite_blob(_isinstance_floats(_core(seed=_FAVAR_SEED)))
    except (ValueError, RuntimeError, FloatingPointError, KeyError, TypeError):
        return {}


def bench_panel_coint() -> dict[str, float]:
    try:
        from quant_fund.models.panel_coint import (
            bench_panel_coint as _core,
        )
    except ImportError:
        return {}
    try:
        return _finite_blob(_isinstance_floats(_core(seed=_PANEL_COINT_SEED)))
    except (ValueError, RuntimeError, FloatingPointError, KeyError, TypeError):
        return {}


def bench_vuong_test() -> dict[str, float]:
    try:
        from quant_fund.models.vuong_test import (
            bench_vuong_test as _core,
        )
    except ImportError:
        return {}
    try:
        return _finite_blob(_isinstance_floats(_core(seed=_VUONG_TEST_SEED)))
    except (ValueError, RuntimeError, FloatingPointError, KeyError, TypeError):
        return {}


def bench_merton_model() -> dict[str, float]:
    try:
        from quant_fund.models.merton_model import (
            bench_merton_model as _core,
        )
    except ImportError:
        return {}
    try:
        return _finite_blob(_isinstance_floats(_core(seed=_MERTON_MODEL_SEED)))
    except (ValueError, RuntimeError, FloatingPointError, KeyError, TypeError):
        return {}


def bench_white_reality() -> dict[str, float]:
    try:
        from quant_fund.metrics.white_reality import (
            bench_white_reality as _core,
        )
    except ImportError:
        return {}
    try:
        return _finite_blob(_isinstance_floats(_core(seed=_WHITE_REALITY_SEED)))
    except (ValueError, RuntimeError, FloatingPointError, KeyError, TypeError):
        return {}
