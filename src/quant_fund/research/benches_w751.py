"""Wave-751 dimer-2 bench adapters (SYNTHETIC-only)."""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.ciucu_dimers import bench_ciucu_dimers
from quant_fund.models.cohn_elkies import bench_cohn_elkies
from quant_fund.models.durfee_arctic import bench_durfee_arctic
from quant_fund.models.karl_dimers import bench_karl_dimers
from quant_fund.models.kassel_kenyon import bench_kassel_kenyon
from quant_fund.models.petrov_dimer import bench_petrov_dimer

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


def bench_kassel_kenyon_family(
    seed: int = _SEED + 14000,
) -> dict[str, float]:
    return _floats(_finite_blob("kassel_kenyon", bench_kassel_kenyon(seed)))


def bench_ciucu_dimers_family(
    seed: int = _SEED + 14001,
) -> dict[str, float]:
    return _floats(_finite_blob("ciucu_dimers", bench_ciucu_dimers(seed)))


def bench_karl_dimers_family(
    seed: int = _SEED + 14002,
) -> dict[str, float]:
    return _floats(_finite_blob("karl_dimers", bench_karl_dimers(seed)))


def bench_petrov_dimer_family(
    seed: int = _SEED + 14003,
) -> dict[str, float]:
    return _floats(_finite_blob("petrov_dimer", bench_petrov_dimer(seed)))


def bench_durfee_arctic_family(
    seed: int = _SEED + 14004,
) -> dict[str, float]:
    return _floats(_finite_blob("durfee_arctic", bench_durfee_arctic(seed)))


def bench_cohn_elkies_family(
    seed: int = _SEED + 14005,
) -> dict[str, float]:
    return _floats(_finite_blob("cohn_elkies", bench_cohn_elkies(seed)))
