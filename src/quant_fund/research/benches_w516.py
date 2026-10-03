"""Wave-516 Kac-Moody/VOA bench adapters (SYNTHETIC-only)."""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.affine_lie import bench_affine_lie
from quant_fund.models.kac_moody import bench_kac_moody
from quant_fund.models.moonshine_module import bench_moonshine_module
from quant_fund.models.vertex_alg import bench_vertex_alg
from quant_fund.models.weyl_kac import bench_weyl_kac
from quant_fund.models.zhu_algebra import bench_zhu_algebra

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


def bench_kac_moody_family(seed: int = _SEED + 3002) -> dict[str, float]:
    return _floats(_finite_blob("kac_moody", bench_kac_moody(seed)))


def bench_weyl_kac_family(seed: int = _SEED + 3003) -> dict[str, float]:
    return _floats(_finite_blob("weyl_kac", bench_weyl_kac(seed)))


def bench_vertex_alg_family(seed: int = _SEED + 3004) -> dict[str, float]:
    return _floats(_finite_blob("vertex_alg", bench_vertex_alg(seed)))


def bench_moonshine_module_family(seed: int = _SEED + 3005) -> dict[str, float]:
    return _floats(_finite_blob("moonshine_module", bench_moonshine_module(seed)))


def bench_affine_lie_family(seed: int = _SEED + 3006) -> dict[str, float]:
    return _floats(_finite_blob("affine_lie", bench_affine_lie(seed)))


def bench_zhu_algebra_family(seed: int = _SEED + 3007) -> dict[str, float]:
    return _floats(_finite_blob("zhu_algebra", bench_zhu_algebra(seed)))
