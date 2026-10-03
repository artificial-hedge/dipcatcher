"""Wave-203 adapters: information-geometry canon — sha256_impl,
aes_sbox, shamir_secret, pedersen_commit, diffie_hellman, ecc_secp256k1 —
benched on SYNTHETIC corpora/generators (torch `nn` extra). Adapters
flatten to a finite float scorecard.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.aes_sbox import bench_aes_sbox
from quant_fund.models.diffie_hellman import bench_diffie_hellman
from quant_fund.models.ecc_secp256k1 import bench_ecc_secp256k1
from quant_fund.models.pedersen_commit import bench_pedersen_commit
from quant_fund.models.sha256_impl import bench_sha256_impl
from quant_fund.models.shamir_secret import bench_shamir_secret

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


def bench_diffie_hellman_family() -> dict[str, float]:
    try:
        return _floats(_finite_blob("diffie_hellman", bench_diffie_hellman(seed=_SEED + 960)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"diffie_hellman bench failed: {exc}") from exc


def bench_sha256_impl_family() -> dict[str, float]:
    try:
        return _floats(_finite_blob("sha256_impl", bench_sha256_impl(seed=_SEED + 961)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"sha256_impl bench failed: {exc}") from exc


def bench_pedersen_commit_family() -> dict[str, float]:
    try:
        return _floats(_finite_blob("pedersen_commit", bench_pedersen_commit(seed=_SEED + 962)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"pedersen_commit bench failed: {exc}") from exc


def bench_aes_sbox_family() -> dict[str, float]:
    try:
        return _floats(_finite_blob("aes_sbox", bench_aes_sbox(seed=_SEED + 963)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"aes_sbox bench failed: {exc}") from exc


def bench_shamir_secret_family() -> dict[str, float]:
    try:
        return _floats(_finite_blob("shamir_secret", bench_shamir_secret(seed=_SEED + 964)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"shamir_secret bench failed: {exc}") from exc


def bench_ecc_secp256k1_family() -> dict[str, float]:
    try:
        return _floats(_finite_blob("ecc_secp256k1", bench_ecc_secp256k1(seed=_SEED + 965)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"ecc_secp256k1 bench failed: {exc}") from exc
