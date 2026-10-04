"""Wave-551 foliation-theory bench adapters (SYNTHETIC-only)."""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.foliation import bench_foliation
from quant_fund.models.godbillon_vey import bench_godbillon_vey
from quant_fund.models.haefliger_struct import bench_haefliger_struct
from quant_fund.models.holonomy_grp import bench_holonomy_grp
from quant_fund.models.novikov_thm import bench_novikov_thm
from quant_fund.models.thurston_fol import bench_thurston_fol

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


def bench_foliation_family(seed: int = _SEED + 3212) -> dict[str, float]:
    return _floats(_finite_blob("foliation", bench_foliation(seed)))


def bench_holonomy_grp_family(seed: int = _SEED + 3213) -> dict[str, float]:
    return _floats(_finite_blob("holonomy_grp", bench_holonomy_grp(seed)))


def bench_godbillon_vey_family(seed: int = _SEED + 3214) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "godbillon_vey",
            bench_godbillon_vey(seed),
        )
    )


def bench_haefliger_struct_family(
    seed: int = _SEED + 3215,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "haefliger_struct",
            bench_haefliger_struct(seed),
        )
    )


def bench_novikov_thm_family(seed: int = _SEED + 3216) -> dict[str, float]:
    return _floats(_finite_blob("novikov_thm", bench_novikov_thm(seed)))


def bench_thurston_fol_family(seed: int = _SEED + 3217) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "thurston_fol",
            bench_thurston_fol(seed),
        )
    )
