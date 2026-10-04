"""Wave-456 pure-motives bench adapters (SYNTHETIC-only)."""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.chow_motive import bench_chow_motive
from quant_fund.models.nori_motive import bench_nori_motive
from quant_fund.models.num_equiv import bench_num_equiv
from quant_fund.models.standard_conj import bench_standard_conj
from quant_fund.models.tate_motive import bench_tate_motive
from quant_fund.models.voev_motive import bench_voev_motive

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


def bench_chow_motive_family(seed: int = _SEED + 2642) -> dict[str, float]:
    return _floats(_finite_blob("chow_motive", bench_chow_motive(seed)))


def bench_nori_motive_family(seed: int = _SEED + 2643) -> dict[str, float]:
    return _floats(_finite_blob("nori_motive", bench_nori_motive(seed)))


def bench_num_equiv_family(seed: int = _SEED + 2644) -> dict[str, float]:
    return _floats(_finite_blob("num_equiv", bench_num_equiv(seed)))


def bench_standard_conj_family(seed: int = _SEED + 2645) -> dict[str, float]:
    return _floats(_finite_blob("standard_conj", bench_standard_conj(seed)))


def bench_voev_motive_family(seed: int = _SEED + 2646) -> dict[str, float]:
    return _floats(_finite_blob("voev_motive", bench_voev_motive(seed)))


def bench_tate_motive_family(seed: int = _SEED + 2647) -> dict[str, float]:
    return _floats(_finite_blob("tate_motive", bench_tate_motive(seed)))
