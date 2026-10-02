"""Wave-69 optional scorecard families — signal decomposition,
multiview statistics, manifold learning, spatial prediction, and
intermittent-demand forecasting. Emitted only when the corresponding
module's `bench_*` self-check completes on its SYNTHETIC fixture.
"""

from __future__ import annotations

import math

_VMD_SEED = 20261231 + 402
_STOCKWELL_SEED = 20261231 + 403
_CCA_SEED = 20261231 + 404
_ISOMAP_SEED = 20261231 + 405
_KRIGING_SEED = 20261231 + 406
_INNOVATIONS_ETS_SEED = 20261231 + 407

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


def bench_vmd() -> dict[str, float]:
    try:
        from quant_fund.models.vmd import bench_vmd as _core
    except ImportError:  # pragma: no cover - optional
        return {}
    try:
        out = _core(seed=_VMD_SEED)
    except (ValueError, RuntimeError, FloatingPointError, KeyError, TypeError):
        return {}
    return _isinstance_floats(out)


def bench_stockwell() -> dict[str, float]:
    try:
        from quant_fund.models.stockwell import bench_stockwell as _core
    except ImportError:  # pragma: no cover - optional
        return {}
    try:
        out = _core(seed=_STOCKWELL_SEED)
    except (ValueError, RuntimeError, FloatingPointError, KeyError, TypeError):
        return {}
    return _isinstance_floats(out)


def bench_cca() -> dict[str, float]:
    try:
        from quant_fund.models.cca import bench_cca as _core
    except ImportError:  # pragma: no cover - optional
        return {}
    try:
        out = _core(seed=_CCA_SEED)
    except (ValueError, RuntimeError, FloatingPointError, KeyError, TypeError):
        return {}
    return _isinstance_floats(out)


def bench_isomap() -> dict[str, float]:
    try:
        from quant_fund.models.isomap import bench_isomap as _core
    except ImportError:  # pragma: no cover - optional
        return {}
    try:
        out = _core(seed=_ISOMAP_SEED)
    except (ValueError, RuntimeError, FloatingPointError, KeyError, TypeError):
        return {}
    return _isinstance_floats(out)


def bench_kriging() -> dict[str, float]:
    try:
        from quant_fund.models.kriging import bench_kriging as _core
    except ImportError:  # pragma: no cover - optional
        return {}
    try:
        out = _core(seed=_KRIGING_SEED)
    except (ValueError, RuntimeError, FloatingPointError, KeyError, TypeError):
        return {}
    return _isinstance_floats(out)


def bench_innovations_ets() -> dict[str, float]:
    try:
        from quant_fund.models.innovations_ets import (
            bench_innovations_ets as _core,
        )
    except ImportError:  # pragma: no cover - optional
        return {}
    try:
        out = _core(seed=_INNOVATIONS_ETS_SEED)
    except (ValueError, RuntimeError, FloatingPointError, KeyError, TypeError):
        return {}
    return _isinstance_floats(out)
