"""Wave-90 optional scorecard families — the
nonparametric-regression and Bayesian-computation
canon: Friedman (1991) MARS multivariate adaptive
regression splines, Breiman-Friedman (1985) ACE and
Tibshirani (1988) AVAS alternating conditional
expectation / variance-stabilizing transforms,
Friedman-Stuetzle (1981) projection-pursuit
regression, Newton-Raftery (1994) harmonic-mean /
Gelfand-Dey (1994) / Chib (1995) / Verdinelli-
Wasserman Savage-Dickey / Ogata (1989)
thermodynamic-integration marginal likelihoods,
Brent (1973)/Ridders (1979)/Illinois regula falsi
1-D root finders and minimizers, and
Belouchrani-Cardoso SOBI (1997)/Cardoso-Souloumiac
JADE (1993)/FOBI (1989) blind source separation.
Emitted only when the corresponding module's
`bench_*` self-check completes on its SYNTHETIC
fixture.
"""

from __future__ import annotations

import math

_MARS_SEED = 20261231 + 528
_ACE_SEED = 20261231 + 529
_PPR_SEED = 20261231 + 530
_ML_SEED = 20261231 + 531
_ROOTS_SEED = 20261231 + 532
_BSS_SEED = 20261231 + 533

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


def bench_mars_regression() -> dict[str, float]:
    try:
        from quant_fund.models.mars_regression import (
            bench_mars as _core,
        )
    except ImportError:  # pragma: no cover - optional
        return {}
    try:
        out = _core(seed=_MARS_SEED)
    except (ValueError, RuntimeError, FloatingPointError, KeyError, TypeError):
        return {}
    return _isinstance_floats(out)


def bench_ace_avas() -> dict[str, float]:
    try:
        from quant_fund.models.ace_avas import (
            bench_ace as _core,
        )
    except ImportError:  # pragma: no cover - optional
        return {}
    try:
        out = _core(seed=_ACE_SEED)
    except (ValueError, RuntimeError, FloatingPointError, KeyError, TypeError):
        return {}
    return _isinstance_floats(out)


def bench_projection_pursuit() -> dict[str, float]:
    try:
        from quant_fund.models.projection_pursuit import (
            bench_ppr as _core,
        )
    except ImportError:  # pragma: no cover - optional
        return {}
    try:
        out = _core(seed=_PPR_SEED)
    except (ValueError, RuntimeError, FloatingPointError, KeyError, TypeError):
        return {}
    return _isinstance_floats(out)


def bench_marginal_likelihood() -> dict[str, float]:
    try:
        from quant_fund.models.marginal_likelihood import (
            bench_ml as _core,
        )
    except ImportError:  # pragma: no cover - optional
        return {}
    try:
        out = _core(seed=_ML_SEED)
    except (ValueError, RuntimeError, FloatingPointError, KeyError, TypeError):
        return {}
    return _isinstance_floats(out)


def bench_root_finders() -> dict[str, float]:
    try:
        from quant_fund.models.root_finders import (
            bench_roots as _core,
        )
    except ImportError:  # pragma: no cover - optional
        return {}
    try:
        out = _core(seed=_ROOTS_SEED)
    except (ValueError, RuntimeError, FloatingPointError, KeyError, TypeError):
        return {}
    return _isinstance_floats(out)


def bench_blind_sources() -> dict[str, float]:
    try:
        from quant_fund.models.blind_sources import (
            bench_bss as _core,
        )
    except ImportError:  # pragma: no cover - optional
        return {}
    try:
        out = _core(seed=_BSS_SEED)
    except (ValueError, RuntimeError, FloatingPointError, KeyError, TypeError):
        return {}
    return _isinstance_floats(out)
