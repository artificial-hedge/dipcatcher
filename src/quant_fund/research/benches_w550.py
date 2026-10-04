"""Wave-550 characteristic-classes bench adapters (SYNTHETIC-only)."""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.chern_character import bench_chern_character
from quant_fund.models.chern_class import bench_chern_class
from quant_fund.models.euler_class import bench_euler_class
from quant_fund.models.hirzebruch_sig import bench_hirzebruch_sig
from quant_fund.models.pontryagin_class import bench_pontryagin_class
from quant_fund.models.todd_genus import bench_todd_genus

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


def bench_chern_class_family(seed: int = _SEED + 3206) -> dict[str, float]:
    return _floats(_finite_blob("chern_class", bench_chern_class(seed)))


def bench_pontryagin_class_family(
    seed: int = _SEED + 3207,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "pontryagin_class",
            bench_pontryagin_class(seed),
        )
    )


def bench_euler_class_family(seed: int = _SEED + 3208) -> dict[str, float]:
    return _floats(_finite_blob("euler_class", bench_euler_class(seed)))


def bench_todd_genus_family(seed: int = _SEED + 3209) -> dict[str, float]:
    return _floats(_finite_blob("todd_genus", bench_todd_genus(seed)))


def bench_chern_character_family(
    seed: int = _SEED + 3210,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "chern_character",
            bench_chern_character(seed),
        )
    )


def bench_hirzebruch_sig_family(seed: int = _SEED + 3211) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "hirzebruch_sig",
            bench_hirzebruch_sig(seed),
        )
    )
