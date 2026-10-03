"""Wave-87 optional scorecard families — Avellaneda-
Stoikov (2008) optimal market making with inventory
control (reservation price, optimal spread, GLFT
intensity), Gillespie (1977) direct SSA + tau-leaping
with SIR/Schlogl reference networks, Hamilton (2018)
regression filter + HP + Baxter-King/Christiano-
Fitzgerald band-pass decomposition, Corwin-Schultz
(2012) high-low spread + Roll (1984) + Amihud (2002)
illiquidity, Fotheringham-Brunsdon-Charlton GWR with
adaptive bandwidth, and Schmittlein/Fader-Hardie-Lee
Pareto/BG-NBD customer lifetime value. Emitted only
when the corresponding module's `bench_*` self-check
completes on its SYNTHETIC fixture.
"""

from __future__ import annotations

import math

_AVELLANEDA_SEED = 20261231 + 510
_GILLESPIE_SEED = 20261231 + 511
_HAMILTON_SEED = 20261231 + 512
_CORWIN_SEED = 20261231 + 513
_GWR_SEED = 20261231 + 514
_PARETO_SEED = 20261231 + 515

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


def bench_avellaneda_stoikov() -> dict[str, float]:
    try:
        from quant_fund.models.avellaneda_stoikov import (
            bench_avellaneda as _core,
        )
    except ImportError:  # pragma: no cover - optional
        return {}
    try:
        out = _core(seed=_AVELLANEDA_SEED)
    except (ValueError, RuntimeError, FloatingPointError, KeyError, TypeError):
        return {}
    return _isinstance_floats(out)


def bench_gillespie_ssa() -> dict[str, float]:
    try:
        from quant_fund.models.gillespie_ssa import (
            bench_gillespie as _core,
        )
    except ImportError:  # pragma: no cover - optional
        return {}
    try:
        out = _core(seed=_GILLESPIE_SEED)
    except (ValueError, RuntimeError, FloatingPointError, KeyError, TypeError):
        return {}
    return _isinstance_floats(out)


def bench_hamilton_filter() -> dict[str, float]:
    try:
        from quant_fund.models.hamilton_filter import (
            bench_hamilton as _core,
        )
    except ImportError:  # pragma: no cover - optional
        return {}
    try:
        out = _core(seed=_HAMILTON_SEED)
    except (ValueError, RuntimeError, FloatingPointError, KeyError, TypeError):
        return {}
    return _isinstance_floats(out)


def bench_corwin_schultz() -> dict[str, float]:
    try:
        from quant_fund.models.corwin_schultz import (
            bench_spread as _core,
        )
    except ImportError:  # pragma: no cover - optional
        return {}
    try:
        out = _core(seed=_CORWIN_SEED)
    except (ValueError, RuntimeError, FloatingPointError, KeyError, TypeError):
        return {}
    return _isinstance_floats(out)


def bench_gwr_spatial() -> dict[str, float]:
    try:
        from quant_fund.models.gwr_spatial import bench_gwr as _core
    except ImportError:  # pragma: no cover - optional
        return {}
    try:
        out = _core(seed=_GWR_SEED)
    except (ValueError, RuntimeError, FloatingPointError, KeyError, TypeError):
        return {}
    return _isinstance_floats(out)


def bench_pareto_nbd() -> dict[str, float]:
    try:
        from quant_fund.models.pareto_nbd import bench_pnbd as _core
    except ImportError:  # pragma: no cover - optional
        return {}
    try:
        out = _core(seed=_PARETO_SEED)
    except (ValueError, RuntimeError, FloatingPointError, KeyError, TypeError):
        return {}
    return _isinstance_floats(out)
