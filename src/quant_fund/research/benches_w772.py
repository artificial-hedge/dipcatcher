"""Wave-772 branching-process bench adapters (SYNTHETIC-only)."""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.branching_imm import bench_branching_imm
from quant_fund.models.crump_mode import bench_crump_mode
from quant_fund.models.galton_watson import bench_galton_watson
from quant_fund.models.kimmel_branch import bench_kimmel_branch
from quant_fund.models.multi_type_branch import (
    bench_multi_type_branch,
)
from quant_fund.models.sevastyanov import bench_sevastyanov

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


def bench_galton_watson_family(
    seed: int = _SEED + 16100,
) -> dict[str, float]:
    return _floats(_finite_blob("galton_watson", bench_galton_watson(seed)))


def bench_branching_imm_family(
    seed: int = _SEED + 16101,
) -> dict[str, float]:
    return _floats(_finite_blob("branching_imm", bench_branching_imm(seed)))


def bench_multi_type_branch_family(
    seed: int = _SEED + 16102,
) -> dict[str, float]:
    return _floats(_finite_blob("multi_type_branch", bench_multi_type_branch(seed)))


def bench_crump_mode_family(
    seed: int = _SEED + 16103,
) -> dict[str, float]:
    return _floats(_finite_blob("crump_mode", bench_crump_mode(seed)))


def bench_kimmel_branch_family(
    seed: int = _SEED + 16104,
) -> dict[str, float]:
    return _floats(_finite_blob("kimmel_branch", bench_kimmel_branch(seed)))


def bench_sevastyanov_family(
    seed: int = _SEED + 16105,
) -> dict[str, float]:
    return _floats(_finite_blob("sevastyanov", bench_sevastyanov(seed)))
