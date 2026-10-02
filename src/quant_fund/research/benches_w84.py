"""Wave-84 optional scorecard families — Hosking (1990)
L-moments (PWM/L-moment estimators + GEV/GLO/GPA/normal
fits, discordancy and heterogeneity statistics),
Saltelli (2010) Sobol first/total-order indices +
Campolongo Morris (1991) elementary effects, Andersen-
Gill (1982) recurrent-event Cox PH (Nelson-Aalen MCF,
PWP gap-time, WLW marginal) with cluster sandwich,
Dawid-Skene (1979) EM crowdsourced annotation +
GLAD (Whitehill 2009), Keogh (2007) matrix-profile
motifs/discords + SAX (Lin 2007), and Hyndman (2011)
MinT hierarchical-forecast reconciliation (OLS/WLS/
LW-shrunk W). Emitted only when the corresponding
module's `bench_*` self-check completes on its
SYNTHETIC fixture.
"""

from __future__ import annotations

import math

_LMOMENTS_SEED = 20261231 + 492
_SOBOL_SENSITIVITY_SEED = 20261231 + 493
_RECURRENT_EVENTS_SEED = 20261231 + 494
_DAWID_SKENE_SEED = 20261231 + 495
_MATRIX_PROFILE_SEED = 20261231 + 496
_HIERARCHICAL_RECONCILIATION_SEED = 20261231 + 497

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


def bench_lmoments() -> dict[str, float]:
    try:
        from quant_fund.models.lmoments import bench_lmoments as _core
    except ImportError:  # pragma: no cover - optional
        return {}
    try:
        out = _core(seed=_LMOMENTS_SEED)
    except (ValueError, RuntimeError, FloatingPointError, KeyError, TypeError):
        return {}
    return _isinstance_floats(out)


def bench_sobol_sensitivity() -> dict[str, float]:
    try:
        from quant_fund.models.sobol_sensitivity import bench_sobol as _core
    except ImportError:  # pragma: no cover - optional
        return {}
    try:
        out = _core(seed=_SOBOL_SENSITIVITY_SEED)
    except (ValueError, RuntimeError, FloatingPointError, KeyError, TypeError):
        return {}
    return _isinstance_floats(out)


def bench_recurrent_events() -> dict[str, float]:
    try:
        from quant_fund.models.recurrent_events import bench_recurrent as _core
    except ImportError:  # pragma: no cover - optional
        return {}
    try:
        out = _core(seed=_RECURRENT_EVENTS_SEED)
    except (ValueError, RuntimeError, FloatingPointError, KeyError, TypeError):
        return {}
    return _isinstance_floats(out)


def bench_dawid_skene() -> dict[str, float]:
    try:
        from quant_fund.models.dawid_skene import bench_dawid_skene as _core
    except ImportError:  # pragma: no cover - optional
        return {}
    try:
        out = _core(seed=_DAWID_SKENE_SEED)
    except (ValueError, RuntimeError, FloatingPointError, KeyError, TypeError):
        return {}
    return _isinstance_floats(out)


def bench_matrix_profile() -> dict[str, float]:
    try:
        from quant_fund.models.matrix_profile import (
            bench_matrix_profile as _core,
        )
    except ImportError:  # pragma: no cover - optional
        return {}
    try:
        out = _core(seed=_MATRIX_PROFILE_SEED)
    except (ValueError, RuntimeError, FloatingPointError, KeyError, TypeError):
        return {}
    return _isinstance_floats(out)


def bench_hierarchical_reconciliation() -> dict[str, float]:
    try:
        from quant_fund.models.hierarchical_reconciliation import (
            bench_reconciliation as _core,
        )
    except ImportError:  # pragma: no cover - optional
        return {}
    try:
        out = _core(seed=_HIERARCHICAL_RECONCILIATION_SEED)
    except (ValueError, RuntimeError, FloatingPointError, KeyError, TypeError):
        return {}
    return _isinstance_floats(out)
