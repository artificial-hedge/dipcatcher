"""Wave-67 adapter benches — optional scorecard families.

Each adapter calls a SYNTHETIC bench on the wave-67 models and
returns a small metric dict consumed by the scorecard runner.
Adapters are honest by contract: they propagate ``score`` and
label every metric ``synthetic_*``. Any import/shape/exception
failure returns ``{}`` so the family is skipped, never faked.

Families (seeds 20261231 + 390..395):
- ``nested_sampling``        — Skilling nested-sampling evidence estimation
- ``smc_samplers``           — Del-Moral tempering SMC posteriors + logZ
- ``synthetic_likelihood``   — Wood/Drovandi Bayesian synthetic likelihood
- ``state_dependent_lp``     — Auerbach-Gorodnichenko state LP impulse responses
- ``moment_inequalities``    — Andrews-Soares GMS moment-inequality test
- ``euler_risk``             — Euler risk contributions (VaR/ES decomposition)
"""

from __future__ import annotations

import math

_SEED = 20261231
_NESTED_SAMPLING_SEED = _SEED + 390
_SMC_SAMPLERS_SEED = _SEED + 391
_SYNTHETIC_LIKELIHOOD_SEED = _SEED + 392
_STATE_DEPENDENT_LP_SEED = _SEED + 393
_MOMENT_INEQUALITIES_SEED = _SEED + 394
_EULER_RISK_SEED = _SEED + 395

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


def bench_nested_sampling() -> dict[str, float]:
    try:
        from quant_fund.models.nested_sampling import (
            bench_nested_sampling as _core,
        )
    except ImportError:  # pragma: no cover - optional
        return {}
    try:
        out = _core(seed=_NESTED_SAMPLING_SEED)
    except (ValueError, RuntimeError, FloatingPointError, KeyError, TypeError):
        return {}
    return _isinstance_floats(out)


def bench_smc_samplers() -> dict[str, float]:
    try:
        from quant_fund.models.smc_samplers import (
            bench_smc_samplers as _core,
        )
    except ImportError:  # pragma: no cover - optional
        return {}
    try:
        out = _core(seed=_SMC_SAMPLERS_SEED)
    except (ValueError, RuntimeError, FloatingPointError, KeyError, TypeError):
        return {}
    return _isinstance_floats(out)


def bench_synthetic_likelihood() -> dict[str, float]:
    try:
        from quant_fund.models.synthetic_likelihood import (
            bench_synthetic_likelihood as _core,
        )
    except ImportError:  # pragma: no cover - optional
        return {}
    try:
        out = _core(seed=_SYNTHETIC_LIKELIHOOD_SEED)
    except (ValueError, RuntimeError, FloatingPointError, KeyError, TypeError):
        return {}
    return _isinstance_floats(out)


def bench_state_dependent_lp() -> dict[str, float]:
    try:
        from quant_fund.models.state_dependent_lp import (
            bench_state_dependent_lp as _core,
        )
    except ImportError:  # pragma: no cover - optional
        return {}
    try:
        out = _core(seed=_STATE_DEPENDENT_LP_SEED)
    except (ValueError, RuntimeError, FloatingPointError, KeyError, TypeError):
        return {}
    return _isinstance_floats(out)


def bench_moment_inequalities() -> dict[str, float]:
    try:
        from quant_fund.models.moment_inequalities import (
            bench_moment_inequalities as _core,
        )
    except ImportError:  # pragma: no cover - optional
        return {}
    try:
        out = _core(seed=_MOMENT_INEQUALITIES_SEED)
    except (ValueError, RuntimeError, FloatingPointError, KeyError, TypeError):
        return {}
    return _isinstance_floats(out)


def bench_euler_risk() -> dict[str, float]:
    try:
        from quant_fund.models.euler_risk import bench_euler_risk as _core
    except ImportError:  # pragma: no cover - optional
        return {}
    try:
        out = _core(seed=_EULER_RISK_SEED)
    except (ValueError, RuntimeError, FloatingPointError, KeyError, TypeError):
        return {}
    return _isinstance_floats(out)
