"""Wave-466 sheaf-3/microlocal bench adapters (SYNTHETIC-only)."""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.kashiwara_schapira import bench_kashiwara_schapira
from quant_fund.models.loc_system import bench_loc_system
from quant_fund.models.micro_supp import bench_micro_supp
from quant_fund.models.perverse_2 import bench_perverse_2
from quant_fund.models.sheaf_homotopy import bench_sheaf_homotopy
from quant_fund.models.stacky_sheaf import bench_stacky_sheaf

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


def bench_micro_supp_family(seed: int = _SEED + 2702) -> dict[str, float]:
    return _floats(_finite_blob("micro_supp", bench_micro_supp(seed)))


def bench_kashiwara_schapira_family(seed: int = _SEED + 2703) -> dict[str, float]:
    return _floats(_finite_blob("kashiwara_schapira", bench_kashiwara_schapira(seed)))


def bench_loc_system_family(seed: int = _SEED + 2704) -> dict[str, float]:
    return _floats(_finite_blob("loc_system", bench_loc_system(seed)))


def bench_perverse_2_family(seed: int = _SEED + 2705) -> dict[str, float]:
    return _floats(_finite_blob("perverse_2", bench_perverse_2(seed)))


def bench_stacky_sheaf_family(seed: int = _SEED + 2706) -> dict[str, float]:
    return _floats(_finite_blob("stacky_sheaf", bench_stacky_sheaf(seed)))


def bench_sheaf_homotopy_family(seed: int = _SEED + 2707) -> dict[str, float]:
    return _floats(_finite_blob("sheaf_homotopy", bench_sheaf_homotopy(seed)))
