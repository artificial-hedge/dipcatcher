"""Wave-683 chromatic-8 bench adapters (SYNTHETIC-only)."""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.blue_shift2 import bench_blue_shift2
from quant_fund.models.chromatic_fracture2 import (
    bench_chromatic_fracture2,
)
from quant_fund.models.fgsl_group2 import bench_fgsl_group2
from quant_fund.models.k_n_local2 import bench_k_n_local2
from quant_fund.models.morava_stabilizer2 import (
    bench_morava_stabilizer2,
)
from quant_fund.models.tate_spec2 import bench_tate_spec2

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


def bench_blue_shift2_family(
    seed: int = _SEED + 7200,
) -> dict[str, float]:
    return _floats(_finite_blob("blue_shift2", bench_blue_shift2(seed)))


def bench_chromatic_fracture2_family(
    seed: int = _SEED + 7201,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "chromatic_fracture2",
            bench_chromatic_fracture2(seed),
        )
    )


def bench_morava_stabilizer2_family(
    seed: int = _SEED + 7202,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "morava_stabilizer2",
            bench_morava_stabilizer2(seed),
        )
    )


def bench_fgsl_group2_family(
    seed: int = _SEED + 7203,
) -> dict[str, float]:
    return _floats(_finite_blob("fgsl_group2", bench_fgsl_group2(seed)))


def bench_tate_spec2_family(
    seed: int = _SEED + 7204,
) -> dict[str, float]:
    return _floats(_finite_blob("tate_spec2", bench_tate_spec2(seed)))


def bench_k_n_local2_family(
    seed: int = _SEED + 7205,
) -> dict[str, float]:
    return _floats(_finite_blob("k_n_local2", bench_k_n_local2(seed)))
