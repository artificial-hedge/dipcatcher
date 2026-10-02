"""Wave-79 optional scorecard families — Hyvarinen FastICA
deflationary independent-component analysis, Reiner-Rubinstein
(1991) closed-form single-barrier option pricing, Schuirmann
two-one-sided-tests equivalence, Robins marginal structural
models via stabilized inverse-probability weights, Lee-Seung
multiplicative-update NMF with consensus cophenetic
diagnostics, and Pitt-Shephard auxiliary particle filtering
for stochastic-volatility state space.
Emitted only when the corresponding module's `bench_*` self-check
completes on its SYNTHETIC fixture.
"""

from __future__ import annotations

import math

_FASTICA_SEED = 20261231 + 462
_BARRIER_OPTIONS_SEED = 20261231 + 463
_TOST_SEED = 20261231 + 464
_MSM_CAUSAL_SEED = 20261231 + 465
_NMF_SEED = 20261231 + 466
_AUXILIARY_PF_SEED = 20261231 + 467

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


def bench_fastica() -> dict[str, float]:
    try:
        from quant_fund.models.fastica import bench_fastica as _core
    except ImportError:  # pragma: no cover - optional
        return {}
    try:
        out = _core(seed=_FASTICA_SEED)
    except (ValueError, RuntimeError, FloatingPointError, KeyError, TypeError):
        return {}
    return _isinstance_floats(out)


def bench_barrier_options() -> dict[str, float]:
    try:
        from quant_fund.models.barrier_options import (
            bench_barrier_options as _core,
        )
    except ImportError:  # pragma: no cover - optional
        return {}
    try:
        out = _core(seed=_BARRIER_OPTIONS_SEED)
    except (ValueError, RuntimeError, FloatingPointError, KeyError, TypeError):
        return {}
    return _isinstance_floats(out)


def bench_tost() -> dict[str, float]:
    try:
        from quant_fund.models.tost import bench_tost as _core
    except ImportError:  # pragma: no cover - optional
        return {}
    try:
        out = _core(seed=_TOST_SEED)
    except (ValueError, RuntimeError, FloatingPointError, KeyError, TypeError):
        return {}
    return _isinstance_floats(out)


def bench_msm_causal() -> dict[str, float]:
    try:
        from quant_fund.models.msm_causal import bench_msm_causal as _core
    except ImportError:  # pragma: no cover - optional
        return {}
    try:
        out = _core(seed=_MSM_CAUSAL_SEED)
    except (ValueError, RuntimeError, FloatingPointError, KeyError, TypeError):
        return {}
    return _isinstance_floats(out)


def bench_nmf() -> dict[str, float]:
    try:
        from quant_fund.models.nmf import bench_nmf as _core
    except ImportError:  # pragma: no cover - optional
        return {}
    try:
        out = _core(seed=_NMF_SEED)
    except (ValueError, RuntimeError, FloatingPointError, KeyError, TypeError):
        return {}
    return _isinstance_floats(out)


def bench_auxiliary_pf() -> dict[str, float]:
    try:
        from quant_fund.models.auxiliary_pf import (
            bench_auxiliary_pf as _core,
        )
    except ImportError:  # pragma: no cover - optional
        return {}
    try:
        out = _core(seed=_AUXILIARY_PF_SEED)
    except (ValueError, RuntimeError, FloatingPointError, KeyError, TypeError):
        return {}
    return _isinstance_floats(out)
