"""Wave-88 optional scorecard families — Tauchen (1986),
Tauchen-Hussey (1991) and Rouwenhorst (1995) Markov
discretization of AR(1), Lomb (1976)/Scargle (1982)/
Press-Rybicki (1989) periodogram for irregular data,
Broomhead-King (1986)/Golyandina (2001) singular
spectrum analysis, Beck-Katz (1995) panel-corrected
standard errors + Parks (1967) FGLS, Quandt (1960)/
Andrews (1993)/Andrews-Ploberger (1994)/Nyblom (1989)
unknown-breakpoint stability tests, and Friedman (1984)
variable-span supersmoother with Cleveland (1979) LOWESS
baseline. Emitted only when the corresponding module's
`bench_*` self-check completes on its SYNTHETIC fixture.
"""

from __future__ import annotations

import math

_MARKOV_SEED = 20261231 + 516
_LOMB_SEED = 20261231 + 517
_SSA_SEED = 20261231 + 518
_PCSE_SEED = 20261231 + 519
_QA_SEED = 20261231 + 520
_SMOOTHER_SEED = 20261231 + 521

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


def bench_markov_discretization() -> dict[str, float]:
    try:
        from quant_fund.models.markov_discretization import (
            bench_markov as _core,
        )
    except ImportError:  # pragma: no cover - optional
        return {}
    try:
        out = _core(seed=_MARKOV_SEED)
    except (ValueError, RuntimeError, FloatingPointError, KeyError, TypeError):
        return {}
    return _isinstance_floats(out)


def bench_lomb_scargle() -> dict[str, float]:
    try:
        from quant_fund.models.lomb_scargle import bench_ls as _core
    except ImportError:  # pragma: no cover - optional
        return {}
    try:
        out = _core(seed=_LOMB_SEED)
    except (ValueError, RuntimeError, FloatingPointError, KeyError, TypeError):
        return {}
    return _isinstance_floats(out)


def bench_singular_spectrum() -> dict[str, float]:
    try:
        from quant_fund.models.singular_spectrum import (
            bench_ssa as _core,
        )
    except ImportError:  # pragma: no cover - optional
        return {}
    try:
        out = _core(seed=_SSA_SEED)
    except (ValueError, RuntimeError, FloatingPointError, KeyError, TypeError):
        return {}
    return _isinstance_floats(out)


def bench_beck_katz() -> dict[str, float]:
    try:
        from quant_fund.models.beck_katz import bench_pcse as _core
    except ImportError:  # pragma: no cover - optional
        return {}
    try:
        out = _core(seed=_PCSE_SEED)
    except (ValueError, RuntimeError, FloatingPointError, KeyError, TypeError):
        return {}
    return _isinstance_floats(out)


def bench_quandt_andrews() -> dict[str, float]:
    try:
        from quant_fund.models.quandt_andrews import bench_qa as _core
    except ImportError:  # pragma: no cover - optional
        return {}
    try:
        out = _core(seed=_QA_SEED)
    except (ValueError, RuntimeError, FloatingPointError, KeyError, TypeError):
        return {}
    return _isinstance_floats(out)


def bench_friedman_supersmoother() -> dict[str, float]:
    try:
        from quant_fund.models.friedman_supersmoother import (
            bench_supersmoother as _core,
        )
    except ImportError:  # pragma: no cover - optional
        return {}
    try:
        out = _core(seed=_SMOOTHER_SEED)
    except (ValueError, RuntimeError, FloatingPointError, KeyError, TypeError):
        return {}
    return _isinstance_floats(out)
