"""Wave-254 adapters: applied-crypto-2 canon — TLS handshake
state machine, HMAC construction, encrypt-then-MAC AEAD,
Merkle–Damgård length extension, PKCS#7 padding, PBKDF2 —
SYNTHETIC benches.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.aead_etm import bench_aead_etm
from quant_fund.models.cbc_padding import bench_cbc_padding
from quant_fund.models.hmac_construct import bench_hmac_construct
from quant_fund.models.merkle_damgard import bench_merkle_damgard
from quant_fund.models.pbkdf2_kdf import bench_pbkdf2_kdf
from quant_fund.models.tls_handshake import bench_tls_handshake

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


def bench_tls_handshake_family(seed: int = _SEED + 1310) -> dict[str, float]:
    return bench_tls_handshake(seed)


def bench_hmac_construct_family(seed: int = _SEED + 1311) -> dict[str, float]:
    return bench_hmac_construct(seed)


def bench_aead_etm_family(seed: int = _SEED + 1312) -> dict[str, float]:
    return bench_aead_etm(seed)


def bench_merkle_damgard_family(seed: int = _SEED + 1313) -> dict[str, float]:
    return bench_merkle_damgard(seed)


def bench_cbc_padding_family(seed: int = _SEED + 1314) -> dict[str, float]:
    return bench_cbc_padding(seed)


def bench_pbkdf2_kdf_family(seed: int = _SEED + 1315) -> dict[str, float]:
    return bench_pbkdf2_kdf(seed)
