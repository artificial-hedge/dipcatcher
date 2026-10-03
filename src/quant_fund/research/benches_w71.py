"""Wave-71 optional scorecard families — item
response theory, latent class analysis, multivariate
omnibus tests, shape alignment, monotone calibration,
and hierarchical portfolio allocation.
Emitted only when the corresponding module's `bench_*` self-check
completes on its SYNTHETIC fixture.
"""

from __future__ import annotations

import math

_ITEM_RESPONSE_SEED = 20261231 + 414
_LATENT_CLASS_SEED = 20261231 + 415
_MANOVA_SEED = 20261231 + 416
_PROCRUSTES_SEED = 20261231 + 417
_ISOTONIC_SEED = 20261231 + 418
_HRP_SEED = 20261231 + 419

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


def bench_item_response() -> dict[str, float]:
    try:
        from quant_fund.models.item_response import (
            bench_item_response as _core,
        )
    except ImportError:  # pragma: no cover - optional
        return {}
    try:
        out = _core(seed=_ITEM_RESPONSE_SEED)
    except (ValueError, RuntimeError, FloatingPointError, KeyError, TypeError):
        return {}
    return _isinstance_floats(out)


def bench_latent_class() -> dict[str, float]:
    try:
        from quant_fund.models.latent_class import (
            bench_latent_class as _core,
        )
    except ImportError:  # pragma: no cover - optional
        return {}
    try:
        out = _core(seed=_LATENT_CLASS_SEED)
    except (ValueError, RuntimeError, FloatingPointError, KeyError, TypeError):
        return {}
    return _isinstance_floats(out)


def bench_manova() -> dict[str, float]:
    try:
        from quant_fund.models.manova import (
            bench_manova as _core,
        )
    except ImportError:  # pragma: no cover - optional
        return {}
    try:
        out = _core(seed=_MANOVA_SEED)
    except (ValueError, RuntimeError, FloatingPointError, KeyError, TypeError):
        return {}
    return _isinstance_floats(out)


def bench_procrustes() -> dict[str, float]:
    try:
        from quant_fund.models.procrustes import bench_procrustes as _core
    except ImportError:  # pragma: no cover - optional
        return {}
    try:
        out = _core(seed=_PROCRUSTES_SEED)
    except (ValueError, RuntimeError, FloatingPointError, KeyError, TypeError):
        return {}
    return _isinstance_floats(out)


def bench_isotonic() -> dict[str, float]:
    try:
        from quant_fund.models.isotonic import (
            bench_isotonic as _core,
        )
    except ImportError:  # pragma: no cover - optional
        return {}
    try:
        out = _core(seed=_ISOTONIC_SEED)
    except (ValueError, RuntimeError, FloatingPointError, KeyError, TypeError):
        return {}
    return _isinstance_floats(out)


def bench_hrp() -> dict[str, float]:
    try:
        from quant_fund.models.hrp import (
            bench_hrp as _core,
        )
    except ImportError:  # pragma: no cover - optional
        return {}
    try:
        out = _core(seed=_HRP_SEED)
    except (ValueError, RuntimeError, FloatingPointError, KeyError, TypeError):
        return {}
    return _isinstance_floats(out)
