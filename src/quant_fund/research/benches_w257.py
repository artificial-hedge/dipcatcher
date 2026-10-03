"""Wave-257 adapters: lattice/advanced-crypto canon — LWE key
exchange, NTRU, BFV FHE, SIS hash, OR-proofs,
Chaum-Pedersen — SYNTHETIC benches.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.bfv_fhe import bench_bfv_fhe
from quant_fund.models.chaum_pedersen import bench_chaum_pedersen
from quant_fund.models.lwe_kex import bench_lwe_kex
from quant_fund.models.ntru_toy import bench_ntru_toy
from quant_fund.models.sigma_or_proof import bench_sigma_or_proof
from quant_fund.models.sis_hash import bench_sis_hash

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


def bench_lwe_kex_family(seed: int = _SEED + 1340) -> dict[str, float]:
    return bench_lwe_kex(seed)


def bench_ntru_toy_family(seed: int = _SEED + 1341) -> dict[str, float]:
    return bench_ntru_toy(seed)


def bench_bfv_fhe_family(seed: int = _SEED + 1342) -> dict[str, float]:
    return bench_bfv_fhe(seed)


def bench_sis_hash_family(seed: int = _SEED + 1343) -> dict[str, float]:
    return bench_sis_hash(seed)


def bench_sigma_or_proof_family(seed: int = _SEED + 1344) -> dict[str, float]:
    return bench_sigma_or_proof(seed)


def bench_chaum_pedersen_family(seed: int = _SEED + 1345) -> dict[str, float]:
    return bench_chaum_pedersen(seed)
