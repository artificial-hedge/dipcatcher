"""Wave-297 post-quantum crypto canon bench adapters (deterministic, SYNTHETIC-only)."""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.dilithium_sig import bench_dilithium_sig
from quant_fund.models.frodokem import bench_frodokem
from quant_fund.models.kyber_kem import bench_kyber_kem
from quant_fund.models.ntt_ring import bench_ntt_ring
from quant_fund.models.sphincs_sig import bench_sphincs_sig
from quant_fund.models.xmss_sig import bench_xmss_sig

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


def bench_ntt_ring_family(seed: int = _SEED + 1688) -> dict[str, float]:
    return _floats(_finite_blob("ntt_ring", bench_ntt_ring(seed)))


def bench_kyber_kem_family(seed: int = _SEED + 1689) -> dict[str, float]:
    return _floats(_finite_blob("kyber_kem", bench_kyber_kem(seed)))


def bench_dilithium_sig_family(seed: int = _SEED + 1690) -> dict[str, float]:
    return _floats(_finite_blob("dilithium_sig", bench_dilithium_sig(seed)))


def bench_frodokem_family(seed: int = _SEED + 1691) -> dict[str, float]:
    return _floats(_finite_blob("frodokem", bench_frodokem(seed)))


def bench_xmss_sig_family(seed: int = _SEED + 1692) -> dict[str, float]:
    return _floats(_finite_blob("xmss_sig", bench_xmss_sig(seed)))


def bench_sphincs_sig_family(seed: int = _SEED + 1693) -> dict[str, float]:
    return _floats(_finite_blob("sphincs_sig", bench_sphincs_sig(seed)))
