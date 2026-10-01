"""Wave-62 adapter benches — optional scorecard families.

Each adapter calls a SYNTHETIC bench on the wave-62 models and
returns a small metric dict consumed by the scorecard runner.
Adapters are honest by contract: they propagate ``score`` and
label every metric ``synthetic_*``. Any import/shape/exception
failure returns ``{}`` so the family is skipped, never faked.

Families (seeds 20261231 + 360..365):
- ``entropy_pooling``  — Meucci min-relative-entropy views
- ``narrative_svar``   — Antolin-Diaz & Rubio-Ramirez narrative SVAR
- ``bfast``            — Verbesselt seasonal+trend break detection
- ``stable_dist``      — Nolan alpha-stable fit (McCulloch + CF-ML)
- ``asian_option``     — Kemna-Vorst geometric + CV arithmetic MC
- ``black_litterman``  — Black-Litterman equilibrium posterior
"""

from __future__ import annotations

import math

_SEED = 20261231
_ENTROPY_POOL_SEED = _SEED + 360
_NARRATIVE_SVAR_SEED = _SEED + 361
_BFAST_SEED = _SEED + 362
_STABLE_DIST_SEED = _SEED + 363
_ASIAN_OPTION_SEED = _SEED + 364
_BLACK_LITTERMAN_SEED = _SEED + 365

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


def bench_entropy_pooling() -> dict[str, float]:
    try:
        from quant_fund.models.entropy_pooling import (
            bench_entropy_pool as _core,
        )
    except ImportError:  # pragma: no cover - optional
        return {}
    try:
        out = _core(seed=_ENTROPY_POOL_SEED)
    except (ValueError, RuntimeError, FloatingPointError, KeyError, TypeError):
        return {}
    return _isinstance_floats(out)


def bench_narrative_svar() -> dict[str, float]:
    try:
        from quant_fund.models.narrative_svar import bench_narrative as _core
    except ImportError:  # pragma: no cover - optional
        return {}
    try:
        out = _core(seed=_NARRATIVE_SVAR_SEED)
    except (ValueError, RuntimeError, FloatingPointError, KeyError, TypeError):
        return {}
    return _isinstance_floats(out)


def bench_bfast() -> dict[str, float]:
    try:
        from quant_fund.models.bfast import bench_bfast as _core
    except ImportError:  # pragma: no cover - optional
        return {}
    try:
        out = _core(seed=_BFAST_SEED)
    except (ValueError, RuntimeError, FloatingPointError, KeyError, TypeError):
        return {}
    return _isinstance_floats(out)


def bench_stable_dist() -> dict[str, float]:
    try:
        from quant_fund.models.stable_dist import bench_stable as _core
    except ImportError:  # pragma: no cover - optional
        return {}
    try:
        out = _core(seed=_STABLE_DIST_SEED)
    except (ValueError, RuntimeError, FloatingPointError, KeyError, TypeError):
        return {}
    return _isinstance_floats(out)


def bench_asian_option() -> dict[str, float]:
    try:
        from quant_fund.models.asian_option import synth_asian as _core
    except ImportError:  # pragma: no cover - optional
        return {}
    try:
        out = _core(seed=_ASIAN_OPTION_SEED)
    except (ValueError, RuntimeError, FloatingPointError, KeyError, TypeError):
        return {}
    return _isinstance_floats(out)


def bench_black_litterman() -> dict[str, float]:
    try:
        from quant_fund.models.black_litterman import bench_bl as _core
    except ImportError:  # pragma: no cover - optional
        return {}
    try:
        out = _core(seed=_BLACK_LITTERMAN_SEED)
    except (ValueError, RuntimeError, FloatingPointError, KeyError, TypeError):
        return {}
    return _isinstance_floats(out)
