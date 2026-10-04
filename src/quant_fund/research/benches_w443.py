"""Wave-443 motivic-2 bench adapters (SYNTHETIC-only)."""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.brauer_grp import bench_brauer_grp
from quant_fund.models.chow_group import bench_chow_group
from quant_fund.models.milnor_conj import bench_milnor_conj
from quant_fund.models.motivic_coh import bench_motivic_coh
from quant_fund.models.motivic_stem import bench_motivic_stem
from quant_fund.models.voevodsky_dm import bench_voevodsky_dm

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


def bench_motivic_coh_family(seed: int = _SEED + 2564) -> dict[str, float]:
    return _floats(_finite_blob("motivic_coh", bench_motivic_coh(seed)))


def bench_chow_group_family(seed: int = _SEED + 2565) -> dict[str, float]:
    return _floats(_finite_blob("chow_group", bench_chow_group(seed)))


def bench_milnor_conj_family(seed: int = _SEED + 2566) -> dict[str, float]:
    return _floats(_finite_blob("milnor_conj", bench_milnor_conj(seed)))


def bench_voevodsky_dm_family(seed: int = _SEED + 2567) -> dict[str, float]:
    return _floats(_finite_blob("voevodsky_dm", bench_voevodsky_dm(seed)))


def bench_motivic_stem_family(seed: int = _SEED + 2568) -> dict[str, float]:
    return _floats(_finite_blob("motivic_stem", bench_motivic_stem(seed)))


def bench_brauer_grp_family(seed: int = _SEED + 2569) -> dict[str, float]:
    return _floats(_finite_blob("brauer_grp", bench_brauer_grp(seed)))
