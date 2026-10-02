"""Wave-81 optional scorecard families — Mosimann/
Minka Dirichlet-multinomial estimation, Banerjee
(2005) von Mises-Fisher mixture EM with Sra kappa
inversion, Freimer-Mudholkar-Kollia-Lin (1988)
generalized lambda quantile matching + starship,
Tukey (1977) g-and-h letter-value estimation,
Robbins-Monro (1951) / Kiefer-Wolfowitz (1952) /
Spall (1992) stochastic approximation, and von
Neumann/Dykstra/Douglas-Rachford projections onto
convex sets.
Emitted only when the corresponding module's `bench_*` self-check
completes on its SYNTHETIC fixture.
"""

from __future__ import annotations

import math

_DIRICHLET_MULTINOMIAL_SEED = 20261231 + 474
_VONMISES_FISHER_SEED = 20261231 + 475
_FKML_SEED = 20261231 + 476
_GANDH_SEED = 20261231 + 477
_ROBBINS_MONRO_SEED = 20261231 + 478
_POCS_SEED = 20261231 + 479

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


def bench_dirichlet_multinomial() -> dict[str, float]:
    try:
        from quant_fund.models.dirichlet_multinomial import (
            bench_dirichlet_multinomial as _core,
        )
    except ImportError:  # pragma: no cover - optional
        return {}
    try:
        out = _core(seed=_DIRICHLET_MULTINOMIAL_SEED)
    except (ValueError, RuntimeError, FloatingPointError, KeyError, TypeError):
        return {}
    return _isinstance_floats(out)


def bench_vonmises_fisher() -> dict[str, float]:
    try:
        from quant_fund.models.vonmises_fisher import (
            bench_vonmises_fisher as _core,
        )
    except ImportError:  # pragma: no cover - optional
        return {}
    try:
        out = _core(seed=_VONMISES_FISHER_SEED)
    except (ValueError, RuntimeError, FloatingPointError, KeyError, TypeError):
        return {}
    return _isinstance_floats(out)


def bench_fkml() -> dict[str, float]:
    try:
        from quant_fund.models.fkml import bench_fkml as _core
    except ImportError:  # pragma: no cover - optional
        return {}
    try:
        out = _core(seed=_FKML_SEED)
    except (ValueError, RuntimeError, FloatingPointError, KeyError, TypeError):
        return {}
    return _isinstance_floats(out)


def bench_gandh() -> dict[str, float]:
    try:
        from quant_fund.models.gandh import bench_gandh as _core
    except ImportError:  # pragma: no cover - optional
        return {}
    try:
        out = _core(seed=_GANDH_SEED)
    except (ValueError, RuntimeError, FloatingPointError, KeyError, TypeError):
        return {}
    return _isinstance_floats(out)


def bench_robbins_monro() -> dict[str, float]:
    try:
        from quant_fund.models.robbins_monro import (
            bench_robbins_monro as _core,
        )
    except ImportError:  # pragma: no cover - optional
        return {}
    try:
        out = _core(seed=_ROBBINS_MONRO_SEED)
    except (ValueError, RuntimeError, FloatingPointError, KeyError, TypeError):
        return {}
    return _isinstance_floats(out)


def bench_pocs() -> dict[str, float]:
    try:
        from quant_fund.models.pocs import bench_pocs as _core
    except ImportError:  # pragma: no cover - optional
        return {}
    try:
        out = _core(seed=_POCS_SEED)
    except (ValueError, RuntimeError, FloatingPointError, KeyError, TypeError):
        return {}
    return _isinstance_floats(out)
