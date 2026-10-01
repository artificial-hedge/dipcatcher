"""Wave-55 optional benchmark adapters.

Each adapter runs one wave-55 module's self-check and returns a
``{metric_key: float}`` blob — all ``synthetic_*`` proper diagnostics, never
market evidence. A lane that is unavailable (ImportError) or whose numbers are
non-finite — the ``_finite_blob`` gate, waves 15-54 precedent — degrades to
``{}`` so the scorecard omits it cleanly.
"""

from __future__ import annotations

import math

_SEED = 20261231  # wave-55 stamp seed

_DIEBOLD_MARIANO_SEED = _SEED + 317
_ENGLE_GRANGER_SEED = _SEED + 318
_GLOSTEN_MILGROM_SEED = _SEED + 319
_HASBROUCK_SEED = _SEED + 320
_BDS_SEED = _SEED + 321
_COCHRANE_PIAZZESI_SEED = _SEED + 322
_ENGLE_NG_SEED = _SEED + 323

_FORBIDDEN_TOKENS = frozenset({"sharpe", "sortino", "calmar", "pnl", "nav"})


def _finite_blob(mapped: dict[str, float]) -> dict[str, float]:
    """``{}`` unless every emitted value is finite (ruff-bench contract)."""
    if all(math.isfinite(v) for v in mapped.values()):
        return mapped
    return {}


def _isinstance_floats(
    raw: dict[str, float] | dict[str, object],
) -> dict[str, float]:
    """Float-coerce a lane blob, dropping str stamps, runtime telemetry,
    and any key carrying a forbidden headline metric token."""
    return {
        k: float(v)
        for k, v in raw.items()
        if isinstance(v, (int, float)) and _FORBIDDEN_TOKENS.isdisjoint(k.lower().split("_"))
    }


def bench_diebold_mariano() -> dict[str, float]:
    try:
        from quant_fund.models.diebold_mariano import (
            bench_diebold_mariano as _core,
        )
    except ImportError:
        return {}
    try:
        return _finite_blob(_isinstance_floats(_core(seed=_DIEBOLD_MARIANO_SEED)))
    except (ValueError, RuntimeError, FloatingPointError, KeyError, TypeError):
        return {}


def bench_engle_granger() -> dict[str, float]:
    try:
        from quant_fund.models.engle_granger import (
            bench_engle_granger as _core,
        )
    except ImportError:
        return {}
    try:
        return _finite_blob(_isinstance_floats(_core(seed=_ENGLE_GRANGER_SEED)))
    except (ValueError, RuntimeError, FloatingPointError, KeyError, TypeError):
        return {}


def bench_glosten_milgrom() -> dict[str, float]:
    try:
        from quant_fund.models.glosten_milgrom import (
            bench_glosten_milgrom as _core,
        )
    except ImportError:
        return {}
    try:
        return _finite_blob(_isinstance_floats(_core(seed=_GLOSTEN_MILGROM_SEED)))
    except (ValueError, RuntimeError, FloatingPointError, KeyError, TypeError):
        return {}


def bench_hasbrouck_is() -> dict[str, float]:
    try:
        from quant_fund.models.hasbrouck_is import (
            bench_hasbrouck as _core,
        )
    except ImportError:
        return {}
    try:
        return _finite_blob(_isinstance_floats(_core(seed=_HASBROUCK_SEED)))
    except (ValueError, RuntimeError, FloatingPointError, KeyError, TypeError):
        return {}


def bench_bds() -> dict[str, float]:
    try:
        from quant_fund.models.bds import bench_bds as _core
    except ImportError:
        return {}
    try:
        return _finite_blob(_isinstance_floats(_core(seed=_BDS_SEED)))
    except (ValueError, RuntimeError, FloatingPointError, KeyError, TypeError):
        return {}


def bench_cochrane_piazzesi() -> dict[str, float]:
    try:
        from quant_fund.models.cochrane_piazzesi import (
            bench_cochrane_piazzesi as _core,
        )
    except ImportError:
        return {}
    try:
        return _finite_blob(_isinstance_floats(_core(seed=_COCHRANE_PIAZZESI_SEED)))
    except (ValueError, RuntimeError, FloatingPointError, KeyError, TypeError):
        return {}


def bench_engle_ng() -> dict[str, float]:
    try:
        from quant_fund.models.engle_ng import bench_engle_ng as _core
    except ImportError:
        return {}
    try:
        return _finite_blob(_isinstance_floats(_core(seed=_ENGLE_NG_SEED)))
    except (ValueError, RuntimeError, FloatingPointError, KeyError, TypeError):
        return {}
