"""Wave-66 adapter benches — optional scorecard families.

Each adapter calls a SYNTHETIC bench on the wave-66 models and
returns a small metric dict consumed by the scorecard runner.
Adapters are honest by contract: they propagate ``score`` and
label every metric ``synthetic_*``. Any import/shape/exception
failure returns ``{}`` so the family is skipped, never faked.

Families (seeds 20261231 + 384..389):
- ``mlmc``                 — multilevel Monte Carlo SDE estimation
- ``black_karasinski``     — trinomial-lattice short-rate pricing
- ``debtrank``             — DebtRank systemic-risk propagation
- ``ews_signals``          — critical-slowing-down early warnings
- ``permutation_entropy``  — Bandt-Pompe ordinal complexity-entropy
- ``svgd``                 — Stein variational gradient descent particles
"""

from __future__ import annotations

import math

_SEED = 20261231
_MLMC_SEED = _SEED + 384
_BLACK_KARASINSKI_SEED = _SEED + 385
_DEBTRANK_SEED = _SEED + 386
_EWS_SIGNALS_SEED = _SEED + 387
_PERMUTATION_ENTROPY_SEED = _SEED + 388
_SVGD_SEED = _SEED + 389

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


def bench_mlmc() -> dict[str, float]:
    try:
        from quant_fund.models.mlmc import bench_mlmc as _core
    except ImportError:  # pragma: no cover - optional
        return {}
    try:
        out = _core(seed=_MLMC_SEED)
    except (ValueError, RuntimeError, FloatingPointError, KeyError, TypeError):
        return {}
    return _isinstance_floats(out)


def bench_black_karasinski() -> dict[str, float]:
    try:
        from quant_fund.models.black_karasinski import (
            bench_black_karasinski as _core,
        )
    except ImportError:  # pragma: no cover - optional
        return {}
    try:
        out = _core(seed=_BLACK_KARASINSKI_SEED)
    except (ValueError, RuntimeError, FloatingPointError, KeyError, TypeError):
        return {}
    return _isinstance_floats(out)


def bench_debtrank() -> dict[str, float]:
    try:
        from quant_fund.models.debtrank import bench_debtrank as _core
    except ImportError:  # pragma: no cover - optional
        return {}
    try:
        out = _core(seed=_DEBTRANK_SEED)
    except (ValueError, RuntimeError, FloatingPointError, KeyError, TypeError):
        return {}
    return _isinstance_floats(out)


def bench_ews_signals() -> dict[str, float]:
    try:
        from quant_fund.models.ews_signals import bench_ews_signals as _core
    except ImportError:  # pragma: no cover - optional
        return {}
    try:
        out = _core(seed=_EWS_SIGNALS_SEED)
    except (ValueError, RuntimeError, FloatingPointError, KeyError, TypeError):
        return {}
    return _isinstance_floats(out)


def bench_permutation_entropy() -> dict[str, float]:
    try:
        from quant_fund.models.permutation_entropy import (
            bench_permutation_entropy as _core,
        )
    except ImportError:  # pragma: no cover - optional
        return {}
    try:
        out = _core(seed=_PERMUTATION_ENTROPY_SEED)
    except (ValueError, RuntimeError, FloatingPointError, KeyError, TypeError):
        return {}
    return _isinstance_floats(out)


def bench_svgd() -> dict[str, float]:
    try:
        from quant_fund.models.svgd import bench_svgd as _core
    except ImportError:  # pragma: no cover - optional
        return {}
    try:
        out = _core(seed=_SVGD_SEED)
    except (ValueError, RuntimeError, FloatingPointError, KeyError, TypeError):
        return {}
    return _isinstance_floats(out)
