"""Wave-650 homotopy-20 bench adapters (SYNTHETIC-only)."""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.anick_htpy import bench_anick_htpy
from quant_fund.models.bousfield_htpy import bench_bousfield_htpy
from quant_fund.models.dror_htpy import bench_dror_htpy
from quant_fund.models.kane_htpy import bench_kane_htpy
from quant_fund.models.moore_htpy import bench_moore_htpy
from quant_fund.models.neisendorfer_htpy import bench_neisendorfer_htpy

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


def bench_bousfield_htpy_family(
    seed: int = _SEED + 3900,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "bousfield_htpy",
            bench_bousfield_htpy(seed),
        )
    )


def bench_dror_htpy_family(
    seed: int = _SEED + 3901,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "dror_htpy",
            bench_dror_htpy(seed),
        )
    )


def bench_kane_htpy_family(
    seed: int = _SEED + 3902,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "kane_htpy",
            bench_kane_htpy(seed),
        )
    )


def bench_moore_htpy_family(
    seed: int = _SEED + 3903,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "moore_htpy",
            bench_moore_htpy(seed),
        )
    )


def bench_neisendorfer_htpy_family(
    seed: int = _SEED + 3904,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "neisendorfer_htpy",
            bench_neisendorfer_htpy(seed),
        )
    )


def bench_anick_htpy_family(
    seed: int = _SEED + 3905,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "anick_htpy",
            bench_anick_htpy(seed),
        )
    )
