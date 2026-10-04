"""Wave-606 algebraic-K-5 bench adapters (SYNTHETIC-only)."""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.connective_k import bench_connective_k
from quant_fund.models.higher_k import bench_higher_k
from quant_fund.models.k_spectrum import bench_k_spectrum
from quant_fund.models.karoubi_k import bench_karoubi_k
from quant_fund.models.nil_k import bench_nil_k
from quant_fund.models.pedersen_weibel import (
    bench_pedersen_weibel,
)

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


def bench_connective_k_family(
    seed: int = _SEED + 3542,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "connective_k",
            bench_connective_k(seed),
        )
    )


def bench_higher_k_family(seed: int = _SEED + 3543) -> dict[str, float]:
    return _floats(_finite_blob("higher_k", bench_higher_k(seed)))


def bench_k_spectrum_family(
    seed: int = _SEED + 3544,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "k_spectrum",
            bench_k_spectrum(seed),
        )
    )


def bench_nil_k_family(seed: int = _SEED + 3545) -> dict[str, float]:
    return _floats(_finite_blob("nil_k", bench_nil_k(seed)))


def bench_karoubi_k_family(
    seed: int = _SEED + 3546,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "karoubi_k",
            bench_karoubi_k(seed),
        )
    )


def bench_pedersen_weibel_family(
    seed: int = _SEED + 3547,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "pedersen_weibel",
            bench_pedersen_weibel(seed),
        )
    )
