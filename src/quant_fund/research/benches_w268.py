"""Wave-268 crypto-4 benches: public-key encryption, ZK proofs, MACs, stream ciphers."""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.chacha_stream import bench_chacha_stream
from quant_fund.models.elgamal_enc import bench_elgamal_enc
from quant_fund.models.fiat_shamir import bench_fiat_shamir
from quant_fund.models.ot_12 import bench_ot_12
from quant_fund.models.paillier_he import bench_paillier_he
from quant_fund.models.poly1305_mac import bench_poly1305_mac

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


def bench_elgamal_enc_family(seed: int = _SEED + 1450) -> dict[str, float]:
    return _floats(_finite_blob("elgamal_enc", bench_elgamal_enc(seed)))


def bench_paillier_he_family(seed: int = _SEED + 1451) -> dict[str, float]:
    return _floats(_finite_blob("paillier_he", bench_paillier_he(seed)))


def bench_fiat_shamir_family(seed: int = _SEED + 1452) -> dict[str, float]:
    return _floats(_finite_blob("fiat_shamir", bench_fiat_shamir(seed)))


def bench_ot_12_family(seed: int = _SEED + 1453) -> dict[str, float]:
    return _floats(_finite_blob("ot_12", bench_ot_12(seed)))


def bench_chacha_stream_family(seed: int = _SEED + 1454) -> dict[str, float]:
    return _floats(_finite_blob("chacha_stream", bench_chacha_stream(seed)))


def bench_poly1305_mac_family(seed: int = _SEED + 1455) -> dict[str, float]:
    return _floats(_finite_blob("poly1305_mac", bench_poly1305_mac(seed)))
