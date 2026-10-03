"""Wave-410 set-theory-3 bench adapters (SYNTHETIC-only)."""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.descriptive3 import bench_descriptive3
from quant_fund.models.forcing2 import bench_forcing2
from quant_fund.models.inner_model import bench_inner_model
from quant_fund.models.ordinal_notation import bench_ordinal_notation
from quant_fund.models.proof_mining import bench_proof_mining
from quant_fund.models.recursion3 import bench_recursion3

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


def bench_forcing2_family(seed: int = _SEED + 2366) -> dict[str, float]:
    return _floats(_finite_blob("forcing2", bench_forcing2(seed)))


def bench_inner_model_family(seed: int = _SEED + 2367) -> dict[str, float]:
    return _floats(_finite_blob("inner_model", bench_inner_model(seed)))


def bench_descriptive3_family(
    seed: int = _SEED + 2368,
) -> dict[str, float]:
    return _floats(_finite_blob("descriptive3", bench_descriptive3(seed)))


def bench_recursion3_family(seed: int = _SEED + 2369) -> dict[str, float]:
    return _floats(_finite_blob("recursion3", bench_recursion3(seed)))


def bench_proof_mining_family(
    seed: int = _SEED + 2370,
) -> dict[str, float]:
    return _floats(_finite_blob("proof_mining", bench_proof_mining(seed)))


def bench_ordinal_notation_family(
    seed: int = _SEED + 2371,
) -> dict[str, float]:
    return _floats(_finite_blob("ordinal_notation", bench_ordinal_notation(seed)))
