"""Wave-120 adapters: cooperative-game + mechanism-design canon —
nucleolus, Banzhaf index, Owen value, Myerson optimal auction,
Groves mechanisms, envy-free allocation — each benched on SYNTHETIC
game instances. Adapters flatten to a finite float scorecard.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.banzhaf import bench_banzhaf
from quant_fund.models.envy_free import bench_envy_free
from quant_fund.models.groves import bench_groves
from quant_fund.models.myerson_auction import bench_myerson
from quant_fund.models.nucleolus import bench_nucleolus
from quant_fund.models.owen import bench_owen

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 20261231

_BENCH_EXC = (ValueError, RuntimeError, FloatingPointError, KeyError, TypeError)


def _finite_blob(name: str, out: dict[str, Any]) -> dict[str, float]:
    flat: dict[str, float] = {}
    for k, v in out.items():
        if any(bad in k.lower() for bad in _FORBIDDEN):
            raise ValueError(f"forbidden metric key {k} in {name}")
        arr = np.asarray(v, dtype=np.float64)
        if arr.ndim == 0:
            f = float(arr)
            if not np.isfinite(f):
                raise ValueError(f"non-finite {k} in {name}")
            flat[k] = f
        else:
            for i, val in enumerate(arr.ravel()):
                f = float(val)
                if not np.isfinite(f):
                    raise ValueError(f"non-finite {k}[{i}] in {name}")
                flat[f"{k}[{i}]"] = f
    return flat


def _floats(out: dict[str, float]) -> dict[str, float]:
    return {k: float(v) for k, v in out.items()}


def bench_nucleolus_family() -> dict[str, float]:
    try:
        return _floats(_finite_blob("nucleolus", bench_nucleolus(seed=_SEED + 708)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"nucleolus bench failed: {exc}") from exc


def bench_banzhaf_family() -> dict[str, float]:
    try:
        return _floats(_finite_blob("banzhaf", bench_banzhaf(seed=_SEED + 709)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"banzhaf bench failed: {exc}") from exc


def bench_owen_family() -> dict[str, float]:
    try:
        return _floats(_finite_blob("owen", bench_owen(seed=_SEED + 710)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"owen bench failed: {exc}") from exc


def bench_myerson_auction_family() -> dict[str, float]:
    try:
        return _floats(_finite_blob("myerson_auction", bench_myerson(seed=_SEED + 711)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"myerson_auction bench failed: {exc}") from exc


def bench_groves_family() -> dict[str, float]:
    try:
        return _floats(_finite_blob("groves", bench_groves(seed=_SEED + 712)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"groves bench failed: {exc}") from exc


def bench_envy_free_family() -> dict[str, float]:
    try:
        return _floats(_finite_blob("envy_free", bench_envy_free(seed=_SEED + 713)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"envy_free bench failed: {exc}") from exc
