"""Wave-506 NIP/distal model-theory bench adapters (SYNTHETIC-only)."""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.distality import bench_distality
from quant_fund.models.dp_rank import bench_dp_rank
from quant_fund.models.forking_seq import bench_forking_seq
from quant_fund.models.honest_def import bench_honest_def
from quant_fund.models.nip_formula import bench_nip_formula
from quant_fund.models.uniform_def import bench_uniform_def

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


def bench_dp_rank_family(seed: int = _SEED + 2942) -> dict[str, float]:
    return _floats(_finite_blob("dp_rank", bench_dp_rank(seed)))


def bench_forking_seq_family(seed: int = _SEED + 2943) -> dict[str, float]:
    return _floats(_finite_blob("forking_seq", bench_forking_seq(seed)))


def bench_honest_def_family(seed: int = _SEED + 2944) -> dict[str, float]:
    return _floats(_finite_blob("honest_def", bench_honest_def(seed)))


def bench_uniform_def_family(seed: int = _SEED + 2945) -> dict[str, float]:
    return _floats(_finite_blob("uniform_def", bench_uniform_def(seed)))


def bench_distality_family(seed: int = _SEED + 2946) -> dict[str, float]:
    return _floats(_finite_blob("distality", bench_distality(seed)))


def bench_nip_formula_family(seed: int = _SEED + 2947) -> dict[str, float]:
    return _floats(_finite_blob("nip_formula", bench_nip_formula(seed)))
