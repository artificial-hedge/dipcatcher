"""Wave-334 secure-computation canon bench adapters (SYNTHETIC-only)."""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.beaver_triple import bench_beaver_triple
from quant_fund.models.bgw_mpc import bench_bgw_mpc
from quant_fund.models.garbled_circuit import bench_garbled_circuit
from quant_fund.models.ot_extension import bench_ot_extension
from quant_fund.models.psi_intersect import bench_psi_intersect
from quant_fund.models.spdz_mac import bench_spdz_mac

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


def bench_garbled_circuit_family(seed: int = _SEED + 1911) -> dict[str, float]:
    return _floats(_finite_blob("garbled_circuit", bench_garbled_circuit(seed)))


def bench_bgw_mpc_family(seed: int = _SEED + 1912) -> dict[str, float]:
    return _floats(_finite_blob("bgw_mpc", bench_bgw_mpc(seed)))


def bench_beaver_triple_family(seed: int = _SEED + 1913) -> dict[str, float]:
    return _floats(_finite_blob("beaver_triple", bench_beaver_triple(seed)))


def bench_ot_extension_family(seed: int = _SEED + 1914) -> dict[str, float]:
    return _floats(_finite_blob("ot_extension", bench_ot_extension(seed)))


def bench_spdz_mac_family(seed: int = _SEED + 1915) -> dict[str, float]:
    return _floats(_finite_blob("spdz_mac", bench_spdz_mac(seed)))


def bench_psi_intersect_family(seed: int = _SEED + 1916) -> dict[str, float]:
    return _floats(_finite_blob("psi_intersect", bench_psi_intersect(seed)))
