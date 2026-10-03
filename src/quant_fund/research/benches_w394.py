"""Wave-394 order-theory/poset canon bench adapters (SYNTHETIC-only)."""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.birkhoff_rep import bench_birkhoff_rep
from quant_fund.models.dilworth_partition import bench_dilworth_partition
from quant_fund.models.downset_lattice import bench_downset_lattice
from quant_fund.models.linear_extension import bench_linear_extension
from quant_fund.models.sperner_bound import bench_sperner_bound
from quant_fund.models.zeta_mobius import bench_zeta_mobius

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


def bench_downset_lattice_family(
    seed: int = _SEED + 2270,
) -> dict[str, float]:
    return _floats(_finite_blob("downset_lattice", bench_downset_lattice(seed)))


def bench_zeta_mobius_family(seed: int = _SEED + 2271) -> dict[str, float]:
    return _floats(_finite_blob("zeta_mobius", bench_zeta_mobius(seed)))


def bench_linear_extension_family(
    seed: int = _SEED + 2272,
) -> dict[str, float]:
    return _floats(_finite_blob("linear_extension", bench_linear_extension(seed)))


def bench_sperner_bound_family(seed: int = _SEED + 2273) -> dict[str, float]:
    return _floats(_finite_blob("sperner_bound", bench_sperner_bound(seed)))


def bench_dilworth_partition_family(
    seed: int = _SEED + 2274,
) -> dict[str, float]:
    return _floats(_finite_blob("dilworth_partition", bench_dilworth_partition(seed)))


def bench_birkhoff_rep_family(seed: int = _SEED + 2275) -> dict[str, float]:
    return _floats(_finite_blob("birkhoff_rep", bench_birkhoff_rep(seed)))
