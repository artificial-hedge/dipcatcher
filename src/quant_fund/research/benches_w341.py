"""Wave-341 modal-logic/topology-2 canon bench adapters (SYNTHETIC-only)."""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.bisimulation import bench_bisimulation
from quant_fund.models.covering_space import bench_covering_space
from quant_fund.models.ef_game import bench_ef_game
from quant_fund.models.fundamental_group import bench_fundamental_group
from quant_fund.models.kripke_semantics import bench_kripke_semantics
from quant_fund.models.topo_separation import bench_topo_separation

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 20261231


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


def bench_kripke_semantics_family(seed: int = _SEED + 1953) -> dict[str, float]:
    return _floats(_finite_blob("kripke_semantics", bench_kripke_semantics(seed)))


def bench_bisimulation_family(seed: int = _SEED + 1954) -> dict[str, float]:
    return _floats(_finite_blob("bisimulation", bench_bisimulation(seed)))


def bench_ef_game_family(seed: int = _SEED + 1955) -> dict[str, float]:
    return _floats(_finite_blob("ef_game", bench_ef_game(seed)))


def bench_fundamental_group_family(seed: int = _SEED + 1956) -> dict[str, float]:
    return _floats(_finite_blob("fundamental_group", bench_fundamental_group(seed)))


def bench_covering_space_family(seed: int = _SEED + 1957) -> dict[str, float]:
    return _floats(_finite_blob("covering_space", bench_covering_space(seed)))


def bench_topo_separation_family(seed: int = _SEED + 1958) -> dict[str, float]:
    return _floats(_finite_blob("topo_separation", bench_topo_separation(seed)))
