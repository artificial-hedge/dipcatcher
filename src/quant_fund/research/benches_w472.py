"""Wave-472 motivic-3 bench adapters (SYNTHETIC-only)."""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.alg_cobordism import bench_alg_cobordism
from quant_fund.models.hermitian_k import bench_hermitian_k
from quant_fund.models.motivic_stem2 import bench_motivic_stem2
from quant_fund.models.oriented_coh import bench_oriented_coh
from quant_fund.models.rostmotive import bench_rostmotive
from quant_fund.models.slice_spec import bench_slice_spec

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


def bench_alg_cobordism_family(seed: int = _SEED + 2738) -> dict[str, float]:
    return _floats(_finite_blob("alg_cobordism", bench_alg_cobordism(seed)))


def bench_hermitian_k_family(seed: int = _SEED + 2739) -> dict[str, float]:
    return _floats(_finite_blob("hermitian_k", bench_hermitian_k(seed)))


def bench_oriented_coh_family(seed: int = _SEED + 2740) -> dict[str, float]:
    return _floats(_finite_blob("oriented_coh", bench_oriented_coh(seed)))


def bench_slice_spec_family(seed: int = _SEED + 2741) -> dict[str, float]:
    return _floats(_finite_blob("slice_spec", bench_slice_spec(seed)))


def bench_motivic_stem2_family(seed: int = _SEED + 2742) -> dict[str, float]:
    return _floats(_finite_blob("motivic_stem2", bench_motivic_stem2(seed)))


def bench_rostmotive_family(seed: int = _SEED + 2743) -> dict[str, float]:
    return _floats(_finite_blob("rostmotive", bench_rostmotive(seed)))
