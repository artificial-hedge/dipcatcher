"""Wave-465 analytic-geometry-3 bench adapters (SYNTHETIC-only)."""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.affinoid_alg import bench_affinoid_alg
from quant_fund.models.dagger_groth import bench_dagger_groth
from quant_fund.models.gauss_point import bench_gauss_point
from quant_fund.models.kedlaya_renorm import bench_kedlaya_renorm
from quant_fund.models.raynaud_gen import bench_raynaud_gen
from quant_fund.models.weierstrass_prep import bench_weierstrass_prep

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


def bench_kedlaya_renorm_family(seed: int = _SEED + 2696) -> dict[str, float]:
    return _floats(_finite_blob("kedlaya_renorm", bench_kedlaya_renorm(seed)))


def bench_dagger_groth_family(seed: int = _SEED + 2697) -> dict[str, float]:
    return _floats(_finite_blob("dagger_groth", bench_dagger_groth(seed)))


def bench_raynaud_gen_family(seed: int = _SEED + 2698) -> dict[str, float]:
    return _floats(_finite_blob("raynaud_gen", bench_raynaud_gen(seed)))


def bench_weierstrass_prep_family(seed: int = _SEED + 2699) -> dict[str, float]:
    return _floats(_finite_blob("weierstrass_prep", bench_weierstrass_prep(seed)))


def bench_gauss_point_family(seed: int = _SEED + 2700) -> dict[str, float]:
    return _floats(_finite_blob("gauss_point", bench_gauss_point(seed)))


def bench_affinoid_alg_family(seed: int = _SEED + 2701) -> dict[str, float]:
    return _floats(_finite_blob("affinoid_alg", bench_affinoid_alg(seed)))
