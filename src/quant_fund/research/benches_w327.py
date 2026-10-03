"""Wave-327 zero-knowledge canon bench adapters (SYNTHETIC-only)."""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.bulletproof_ip import bench_bulletproof_ip
from quant_fund.models.kzg_commit import bench_kzg_commit
from quant_fund.models.plonkish_gate import bench_plonkish_gate
from quant_fund.models.qap_encode import bench_qap_encode
from quant_fund.models.r1cs_check import bench_r1cs_check
from quant_fund.models.snark_circuit import bench_snark_circuit

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


def bench_r1cs_check_family(seed: int = _SEED + 1869) -> dict[str, float]:
    return _floats(_finite_blob("r1cs_check", bench_r1cs_check(seed)))


def bench_qap_encode_family(seed: int = _SEED + 1870) -> dict[str, float]:
    return _floats(_finite_blob("qap_encode", bench_qap_encode(seed)))


def bench_kzg_commit_family(seed: int = _SEED + 1871) -> dict[str, float]:
    return _floats(_finite_blob("kzg_commit", bench_kzg_commit(seed)))


def bench_bulletproof_ip_family(seed: int = _SEED + 1872) -> dict[str, float]:
    return _floats(_finite_blob("bulletproof_ip", bench_bulletproof_ip(seed)))


def bench_plonkish_gate_family(seed: int = _SEED + 1873) -> dict[str, float]:
    return _floats(_finite_blob("plonkish_gate", bench_plonkish_gate(seed)))


def bench_snark_circuit_family(seed: int = _SEED + 1874) -> dict[str, float]:
    return _floats(_finite_blob("snark_circuit", bench_snark_circuit(seed)))
