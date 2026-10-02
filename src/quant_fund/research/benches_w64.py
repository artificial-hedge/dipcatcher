"""Wave-64 adapter benches — optional scorecard families.

Each adapter calls a SYNTHETIC bench on the wave-64 models and
returns a small metric dict consumed by the scorecard runner.
Adapters are honest by contract: they propagate ``score`` and
label every metric ``synthetic_*``. Any import/shape/exception
failure returns ``{}`` so the family is skipped, never faked.

Families (seeds 20261231 + 372..377):
- ``saddlepoint``       — Lugannani-Rice tails + saddlepoint density
- ``mutual_info``       — Kraskov-Stogbauer-Grassberger MI estimator
- ``transfer_entropy``  — Schreiber transfer entropy (directional)
- ``brownian_bridge``   — bridge moments + barrier hit refinement
- ``jarrow_turnbull``   — reduced-form credit: hazard bootstrap + CDS
- ``campbell_shiller``  — log-linear VAR return variance decomposition
"""

from __future__ import annotations

import math

_SEED = 20261231
_SADDLEPOINT_SEED = _SEED + 372
_MUTUAL_INFO_SEED = _SEED + 373
_TRANSFER_ENTROPY_SEED = _SEED + 374
_BROWNIAN_BRIDGE_SEED = _SEED + 375
_JARROW_TURNBULL_SEED = _SEED + 376
_CAMPBELL_SHILLER_SEED = _SEED + 377

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


def bench_saddlepoint() -> dict[str, float]:
    try:
        from quant_fund.models.saddlepoint import bench_saddlepoint as _core
    except ImportError:  # pragma: no cover - optional
        return {}
    try:
        out = _core(seed=_SADDLEPOINT_SEED)
    except (ValueError, RuntimeError, FloatingPointError, KeyError, TypeError):
        return {}
    return _isinstance_floats(out)


def bench_mutual_info() -> dict[str, float]:
    try:
        from quant_fund.models.mutual_info import bench_mutual_info as _core
    except ImportError:  # pragma: no cover - optional
        return {}
    try:
        out = _core(seed=_MUTUAL_INFO_SEED)
    except (ValueError, RuntimeError, FloatingPointError, KeyError, TypeError):
        return {}
    return _isinstance_floats(out)


def bench_transfer_entropy() -> dict[str, float]:
    try:
        from quant_fund.models.transfer_entropy import (
            bench_transfer_entropy as _core,
        )
    except ImportError:  # pragma: no cover - optional
        return {}
    try:
        out = _core(seed=_TRANSFER_ENTROPY_SEED)
    except (ValueError, RuntimeError, FloatingPointError, KeyError, TypeError):
        return {}
    return _isinstance_floats(out)


def bench_brownian_bridge() -> dict[str, float]:
    try:
        from quant_fund.models.brownian_bridge import (
            bench_brownian_bridge as _core,
        )
    except ImportError:  # pragma: no cover - optional
        return {}
    try:
        out = _core(seed=_BROWNIAN_BRIDGE_SEED)
    except (ValueError, RuntimeError, FloatingPointError, KeyError, TypeError):
        return {}
    return _isinstance_floats(out)


def bench_jarrow_turnbull() -> dict[str, float]:
    try:
        from quant_fund.models.jarrow_turnbull import (
            bench_jarrow_turnbull as _core,
        )
    except ImportError:  # pragma: no cover - optional
        return {}
    try:
        out = _core(seed=_JARROW_TURNBULL_SEED)
    except (ValueError, RuntimeError, FloatingPointError, KeyError, TypeError):
        return {}
    return _isinstance_floats(out)


def bench_campbell_shiller() -> dict[str, float]:
    try:
        from quant_fund.models.campbell_shiller import (
            bench_campbell_shiller as _core,
        )
    except ImportError:  # pragma: no cover - optional
        return {}
    try:
        out = _core(seed=_CAMPBELL_SHILLER_SEED)
    except (ValueError, RuntimeError, FloatingPointError, KeyError, TypeError):
        return {}
    return _isinstance_floats(out)
