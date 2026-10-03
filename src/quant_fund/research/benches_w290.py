"""Wave-290 chem-informatics canon bench adapters (deterministic, SYNTHETIC-only)."""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.mol_descriptors import bench_mol_descriptors
from quant_fund.models.morgan_fp import bench_morgan_fp
from quant_fund.models.ring_detect import bench_ring_detect
from quant_fund.models.smiles_parse import bench_smiles_parse
from quant_fund.models.substruct import bench_substruct
from quant_fund.models.tanimoto import bench_tanimoto

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 20261231

_BENCH_EXC = (ValueError, RuntimeError, FloatingPointError, KeyError, TypeError)


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


def bench_smiles_parse_family(seed: int = _SEED + 1646) -> dict[str, float]:
    return _floats(_finite_blob("smiles_parse", bench_smiles_parse(seed)))


def bench_morgan_fp_family(seed: int = _SEED + 1647) -> dict[str, float]:
    return _floats(_finite_blob("morgan_fp", bench_morgan_fp(seed)))


def bench_tanimoto_family(seed: int = _SEED + 1648) -> dict[str, float]:
    return _floats(_finite_blob("tanimoto", bench_tanimoto(seed)))


def bench_mol_descriptors_family(seed: int = _SEED + 1649) -> dict[str, float]:
    return _floats(_finite_blob("mol_descriptors", bench_mol_descriptors(seed)))


def bench_substruct_family(seed: int = _SEED + 1650) -> dict[str, float]:
    return _floats(_finite_blob("substruct", bench_substruct(seed)))


def bench_ring_detect_family(seed: int = _SEED + 1651) -> dict[str, float]:
    return _floats(_finite_blob("ring_detect", bench_ring_detect(seed)))
