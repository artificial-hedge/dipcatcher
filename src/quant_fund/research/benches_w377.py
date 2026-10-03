"""Wave-377 operad canon bench adapters (SYNTHETIC-only)."""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.endomorphism_op import bench_endomorphism_op
from quant_fund.models.little_discs import bench_little_discs
from quant_fund.models.may_recognition import bench_may_recognition
from quant_fund.models.operad_assoc import bench_operad_assoc
from quant_fund.models.operad_comm import bench_operad_comm
from quant_fund.models.operad_tree import bench_operad_tree

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


def bench_operad_assoc_family(seed: int = _SEED + 2168) -> dict[str, float]:
    return _floats(_finite_blob("operad_assoc", bench_operad_assoc(seed)))


def bench_operad_comm_family(seed: int = _SEED + 2169) -> dict[str, float]:
    return _floats(_finite_blob("operad_comm", bench_operad_comm(seed)))


def bench_little_discs_family(seed: int = _SEED + 2170) -> dict[str, float]:
    return _floats(_finite_blob("little_discs", bench_little_discs(seed)))


def bench_operad_tree_family(seed: int = _SEED + 2171) -> dict[str, float]:
    return _floats(_finite_blob("operad_tree", bench_operad_tree(seed)))


def bench_endomorphism_op_family(seed: int = _SEED + 2172) -> dict[str, float]:
    return _floats(_finite_blob("endomorphism_op", bench_endomorphism_op(seed)))


def bench_may_recognition_family(seed: int = _SEED + 2173) -> dict[str, float]:
    return _floats(_finite_blob("may_recognition", bench_may_recognition(seed)))
