"""Wave-240 adapters: applied-crypto canon — toy RSA, Winternitz OTS,
Merkle signatures, Chaum blind sigs, Schnorr sigma protocol, hash
commitments — SYNTHETIC correctness benches.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.blind_sig import bench_blind_sig
from quant_fund.models.commit_reveal import bench_commit_reveal
from quant_fund.models.merkle_ots import bench_merkle_ots
from quant_fund.models.rsa_toy import bench_rsa_toy
from quant_fund.models.winternitz_ots import bench_winternitz_ots
from quant_fund.models.zkp_schnorr import bench_zkp_schnorr

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


def bench_blind_sig_family() -> dict[str, float]:
    try:
        return _floats(_finite_blob("blind_sig", bench_blind_sig(seed=_SEED + 1170)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"blind_sig bench failed: {exc}") from exc


def bench_commit_reveal_family() -> dict[str, float]:
    try:
        return _floats(_finite_blob("commit_reveal", bench_commit_reveal(seed=_SEED + 1171)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"commit_reveal bench failed: {exc}") from exc


def bench_merkle_ots_family() -> dict[str, float]:
    try:
        return _floats(_finite_blob("merkle_ots", bench_merkle_ots(seed=_SEED + 1172)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"merkle_ots bench failed: {exc}") from exc


def bench_rsa_toy_family() -> dict[str, float]:
    try:
        return _floats(_finite_blob("rsa_toy", bench_rsa_toy(seed=_SEED + 1173)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"rsa_toy bench failed: {exc}") from exc


def bench_winternitz_ots_family() -> dict[str, float]:
    try:
        return _floats(_finite_blob("winternitz_ots", bench_winternitz_ots(seed=_SEED + 1174)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"winternitz_ots bench failed: {exc}") from exc


def bench_zkp_schnorr_family() -> dict[str, float]:
    try:
        return _floats(_finite_blob("zkp_schnorr", bench_zkp_schnorr(seed=_SEED + 1175)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"zkp_schnorr bench failed: {exc}") from exc
