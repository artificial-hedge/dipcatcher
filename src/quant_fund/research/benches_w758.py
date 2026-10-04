"""Wave-758 GFF-2 bench adapters (SYNTHETIC-only)."""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.aru_gff import bench_aru_gff
from quant_fund.models.bolthausen_gff import bench_bolthausen_gff
from quant_fund.models.chatterjee_gff import bench_chatterjee_gff
from quant_fund.models.ding_zeitouni import bench_ding_zeitouni
from quant_fund.models.najafi_gff import bench_najafi_gff
from quant_fund.models.powell_gff import bench_powell_gff

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


def bench_powell_gff_family(
    seed: int = _SEED + 14700,
) -> dict[str, float]:
    return _floats(_finite_blob("powell_gff", bench_powell_gff(seed)))


def bench_aru_gff_family(
    seed: int = _SEED + 14701,
) -> dict[str, float]:
    return _floats(_finite_blob("aru_gff", bench_aru_gff(seed)))


def bench_ding_zeitouni_family(
    seed: int = _SEED + 14702,
) -> dict[str, float]:
    return _floats(_finite_blob("ding_zeitouni", bench_ding_zeitouni(seed)))


def bench_chatterjee_gff_family(
    seed: int = _SEED + 14703,
) -> dict[str, float]:
    return _floats(_finite_blob("chatterjee_gff", bench_chatterjee_gff(seed)))


def bench_bolthausen_gff_family(
    seed: int = _SEED + 14704,
) -> dict[str, float]:
    return _floats(_finite_blob("bolthausen_gff", bench_bolthausen_gff(seed)))


def bench_najafi_gff_family(
    seed: int = _SEED + 14705,
) -> dict[str, float]:
    return _floats(_finite_blob("najafi_gff", bench_najafi_gff(seed)))
