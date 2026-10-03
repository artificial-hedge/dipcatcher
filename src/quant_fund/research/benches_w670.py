"""Wave-670 chromatic-7 bench adapters (SYNTHETIC-only)."""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.chromatic_l3 import bench_chromatic_l3
from quant_fund.models.morava_e2 import bench_morava_e2
from quant_fund.models.morava_k3 import bench_morava_k3
from quant_fund.models.picard_spec2 import bench_picard_spec2
from quant_fund.models.red_shift2 import bench_red_shift2
from quant_fund.models.telescope_tower3 import (
    bench_telescope_tower3,
)

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


def bench_morava_k3_family(
    seed: int = _SEED + 5900,
) -> dict[str, float]:
    return _floats(_finite_blob("morava_k3", bench_morava_k3(seed)))


def bench_morava_e2_family(
    seed: int = _SEED + 5901,
) -> dict[str, float]:
    return _floats(_finite_blob("morava_e2", bench_morava_e2(seed)))


def bench_chromatic_l3_family(
    seed: int = _SEED + 5902,
) -> dict[str, float]:
    return _floats(_finite_blob("chromatic_l3", bench_chromatic_l3(seed)))


def bench_telescope_tower3_family(
    seed: int = _SEED + 5903,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "telescope_tower3",
            bench_telescope_tower3(seed),
        )
    )


def bench_picard_spec2_family(
    seed: int = _SEED + 5904,
) -> dict[str, float]:
    return _floats(_finite_blob("picard_spec2", bench_picard_spec2(seed)))


def bench_red_shift2_family(
    seed: int = _SEED + 5905,
) -> dict[str, float]:
    return _floats(_finite_blob("red_shift2", bench_red_shift2(seed)))
