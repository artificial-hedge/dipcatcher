"""Wave-350 representation-theory-2 canon bench adapters (SYNTHETIC-only)."""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.character_table_s3 import bench_character_table_s3
from quant_fund.models.fourier_sn import bench_fourier_sn
from quant_fund.models.induced_rep import bench_induced_rep
from quant_fund.models.perm_rep import bench_perm_rep
from quant_fund.models.regular_rep import bench_regular_rep
from quant_fund.models.schur_ortho import bench_schur_ortho

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


def bench_character_table_s3_family(seed: int = _SEED + 2007) -> dict[str, float]:
    return _floats(_finite_blob("character_table_s3", bench_character_table_s3(seed)))


def bench_perm_rep_family(seed: int = _SEED + 2008) -> dict[str, float]:
    return _floats(_finite_blob("perm_rep", bench_perm_rep(seed)))


def bench_schur_ortho_family(seed: int = _SEED + 2009) -> dict[str, float]:
    return _floats(_finite_blob("schur_ortho", bench_schur_ortho(seed)))


def bench_induced_rep_family(seed: int = _SEED + 2010) -> dict[str, float]:
    return _floats(_finite_blob("induced_rep", bench_induced_rep(seed)))


def bench_fourier_sn_family(seed: int = _SEED + 2011) -> dict[str, float]:
    return _floats(_finite_blob("fourier_sn", bench_fourier_sn(seed)))


def bench_regular_rep_family(seed: int = _SEED + 2012) -> dict[str, float]:
    return _floats(_finite_blob("regular_rep", bench_regular_rep(seed)))
