"""Wave-748 vertex-model bench adapters (SYNTHETIC-only)."""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.aggarwal_sixv import bench_aggarwal_sixv
from quant_fund.models.baxter_vertex import bench_baxter_vertex
from quant_fund.models.borodin_sixv import bench_borodin_sixv
from quant_fund.models.corwin_petrov import bench_corwin_petrov
from quant_fund.models.gowers_knot import bench_gowers_knot
from quant_fund.models.reshetikhin_vertex import (
    bench_reshetikhin_vertex,
)

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


def bench_borodin_sixv_family(
    seed: int = _SEED + 13700,
) -> dict[str, float]:
    return _floats(_finite_blob("borodin_sixv", bench_borodin_sixv(seed)))


def bench_gowers_knot_family(
    seed: int = _SEED + 13701,
) -> dict[str, float]:
    return _floats(_finite_blob("gowers_knot", bench_gowers_knot(seed)))


def bench_baxter_vertex_family(
    seed: int = _SEED + 13702,
) -> dict[str, float]:
    return _floats(_finite_blob("baxter_vertex", bench_baxter_vertex(seed)))


def bench_reshetikhin_vertex_family(
    seed: int = _SEED + 13703,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "reshetikhin_vertex",
            bench_reshetikhin_vertex(seed),
        )
    )


def bench_corwin_petrov_family(
    seed: int = _SEED + 13704,
) -> dict[str, float]:
    return _floats(_finite_blob("corwin_petrov", bench_corwin_petrov(seed)))


def bench_aggarwal_sixv_family(
    seed: int = _SEED + 13705,
) -> dict[str, float]:
    return _floats(_finite_blob("aggarwal_sixv", bench_aggarwal_sixv(seed)))
