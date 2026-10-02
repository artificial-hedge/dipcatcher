"""Wave-232 adapters: blockchain/consensus canon — Merkle proofs,
proof-of-work, UTXO validation, difficulty retarget, fork resolution,
block validation — SYNTHETIC protocol benches.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.block_validator import bench_block_validator
from quant_fund.models.difficulty_retarget import bench_difficulty_retarget
from quant_fund.models.fork_resolution import bench_fork_resolution
from quant_fund.models.merkle_tree import bench_merkle_tree
from quant_fund.models.proof_of_work import bench_proof_of_work
from quant_fund.models.utxo_set import bench_utxo_set

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


def bench_block_validator_family() -> dict[str, float]:
    try:
        return _floats(_finite_blob("block_validator", bench_block_validator(seed=_SEED + 1090)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"block_validator bench failed: {exc}") from exc


def bench_difficulty_retarget_family() -> dict[str, float]:
    try:
        return _floats(
            _finite_blob("difficulty_retarget", bench_difficulty_retarget(seed=_SEED + 1091))
        )
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"difficulty_retarget bench failed: {exc}") from exc


def bench_fork_resolution_family() -> dict[str, float]:
    try:
        return _floats(_finite_blob("fork_resolution", bench_fork_resolution(seed=_SEED + 1092)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"fork_resolution bench failed: {exc}") from exc


def bench_merkle_tree_family() -> dict[str, float]:
    try:
        return _floats(_finite_blob("merkle_tree", bench_merkle_tree(seed=_SEED + 1093)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"merkle_tree bench failed: {exc}") from exc


def bench_proof_of_work_family() -> dict[str, float]:
    try:
        return _floats(_finite_blob("proof_of_work", bench_proof_of_work(seed=_SEED + 1094)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"proof_of_work bench failed: {exc}") from exc


def bench_utxo_set_family() -> dict[str, float]:
    try:
        return _floats(_finite_blob("utxo_set", bench_utxo_set(seed=_SEED + 1095)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"utxo_set bench failed: {exc}") from exc
