"""Wave-86 optional scorecard families — ASTM E1049-85
rainflow cycle counting + Palmgren-Miner damage with
Goodman/Gerber/SWT mean-stress corrections, Bayesian
multi-model tracking (Blom & Bar-Shalom 1988 IMM +
Bar-Shalom & Tse 1975 PDA), Karrer-Newman (2011)
degree-corrected SBM inference via regularized spectral
clustering (Amini 2013) with Danon (2005) NMI, positive-
unlabeled learning (Elkan-Noto 2008 + du Plessis 2014
unbiased risk + Kiryo 2017 nnPU), GR4J rainfall-runoff
(Perrin 2003) + Muskingum (1938) routing with NSE/KGE,
and Brinson-Hood-Beebower (1986) / Carino (1999)
performance attribution. Emitted only when the
corresponding module's `bench_*` self-check completes on
its SYNTHETIC fixture.
"""

from __future__ import annotations

import math

_RAINFLOW_SEED = 20261231 + 504
_TRACKING_SEED = 20261231 + 505
_SBM_SEED = 20261231 + 506
_PU_SEED = 20261231 + 507
_HYDRO_SEED = 20261231 + 508
_BRINSON_SEED = 20261231 + 509

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


def bench_rainflow_fatigue() -> dict[str, float]:
    try:
        from quant_fund.models.rainflow_fatigue import (
            bench_rainflow as _core,
        )
    except ImportError:  # pragma: no cover - optional
        return {}
    try:
        out = _core(seed=_RAINFLOW_SEED)
    except (ValueError, RuntimeError, FloatingPointError, KeyError, TypeError):
        return {}
    return _isinstance_floats(out)


def bench_bayesian_tracking() -> dict[str, float]:
    try:
        from quant_fund.models.bayesian_tracking import (
            bench_tracking as _core,
        )
    except ImportError:  # pragma: no cover - optional
        return {}
    try:
        out = _core(seed=_TRACKING_SEED)
    except (ValueError, RuntimeError, FloatingPointError, KeyError, TypeError):
        return {}
    return _isinstance_floats(out)


def bench_sbm_inference() -> dict[str, float]:
    try:
        from quant_fund.models.sbm_inference import bench_sbm as _core
    except ImportError:  # pragma: no cover - optional
        return {}
    try:
        out = _core(seed=_SBM_SEED)
    except (ValueError, RuntimeError, FloatingPointError, KeyError, TypeError):
        return {}
    return _isinstance_floats(out)


def bench_pu_learning() -> dict[str, float]:
    try:
        from quant_fund.models.pu_learning import bench_pu as _core
    except ImportError:  # pragma: no cover - optional
        return {}
    try:
        out = _core(seed=_PU_SEED)
    except (ValueError, RuntimeError, FloatingPointError, KeyError, TypeError):
        return {}
    return _isinstance_floats(out)


def bench_gr4j_hydrology() -> dict[str, float]:
    try:
        from quant_fund.models.gr4j_hydrology import (
            bench_hydrology as _core,
        )
    except ImportError:  # pragma: no cover - optional
        return {}
    try:
        out = _core(seed=_HYDRO_SEED)
    except (ValueError, RuntimeError, FloatingPointError, KeyError, TypeError):
        return {}
    return _isinstance_floats(out)


def bench_brinson_attribution() -> dict[str, float]:
    try:
        from quant_fund.models.brinson_attribution import (
            bench_brinson as _core,
        )
    except ImportError:  # pragma: no cover - optional
        return {}
    try:
        out = _core(seed=_BRINSON_SEED)
    except (ValueError, RuntimeError, FloatingPointError, KeyError, TypeError):
        return {}
    return _isinstance_floats(out)
