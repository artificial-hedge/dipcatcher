"""Wave-468 model-theory-7 bench adapters (SYNTHETIC-only)."""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.abstract_erc import bench_abstract_erc
from quant_fund.models.nip_theory import bench_nip_theory
from quant_fund.models.nonforking import bench_nonforking
from quant_fund.models.o_minimal import bench_o_minimal
from quant_fund.models.simple_theory import bench_simple_theory
from quant_fund.models.tame_metric import bench_tame_metric

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


def bench_o_minimal_family(seed: int = _SEED + 2714) -> dict[str, float]:
    return _floats(_finite_blob("o_minimal", bench_o_minimal(seed)))


def bench_nip_theory_family(seed: int = _SEED + 2715) -> dict[str, float]:
    return _floats(_finite_blob("nip_theory", bench_nip_theory(seed)))


def bench_nonforking_family(seed: int = _SEED + 2716) -> dict[str, float]:
    return _floats(_finite_blob("nonforking", bench_nonforking(seed)))


def bench_simple_theory_family(seed: int = _SEED + 2717) -> dict[str, float]:
    return _floats(_finite_blob("simple_theory", bench_simple_theory(seed)))


def bench_abstract_erc_family(seed: int = _SEED + 2718) -> dict[str, float]:
    return _floats(_finite_blob("abstract_erc", bench_abstract_erc(seed)))


def bench_tame_metric_family(seed: int = _SEED + 2719) -> dict[str, float]:
    return _floats(_finite_blob("tame_metric", bench_tame_metric(seed)))
