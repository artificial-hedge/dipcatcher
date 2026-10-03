"""Wave-636 tensor-category-3 bench adapters (SYNTHETIC-only)."""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.ek_subfactor import (
    bench_ek_subfactor,
)
from quant_fund.models.gyro_cat import bench_gyro_cat
from quant_fund.models.haagerup_sub import (
    bench_haagerup_sub,
)
from quant_fund.models.sovereign_cat import (
    bench_sovereign_cat,
)
from quant_fund.models.sylleptic import bench_sylleptic
from quant_fund.models.yang_lee_cat import (
    bench_yang_lee_cat,
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


def bench_sylleptic_family(
    seed: int = _SEED + 3722,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "sylleptic",
            bench_sylleptic(seed),
        )
    )


def bench_haagerup_sub_family(
    seed: int = _SEED + 3723,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "haagerup_sub",
            bench_haagerup_sub(seed),
        )
    )


def bench_ek_subfactor_family(
    seed: int = _SEED + 3724,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "ek_subfactor",
            bench_ek_subfactor(seed),
        )
    )


def bench_gyro_cat_family(
    seed: int = _SEED + 3725,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "gyro_cat",
            bench_gyro_cat(seed),
        )
    )


def bench_yang_lee_cat_family(
    seed: int = _SEED + 3726,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "yang_lee_cat",
            bench_yang_lee_cat(seed),
        )
    )


def bench_sovereign_cat_family(
    seed: int = _SEED + 3727,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "sovereign_cat",
            bench_sovereign_cat(seed),
        )
    )
