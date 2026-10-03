"""Wave-638 algebraic-K-7 bench adapters (SYNTHETIC-only)."""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.fundamental_cat import (
    bench_fundamental_cat,
)
from quant_fund.models.grayson_s import bench_grayson_s
from quant_fund.models.karoubi_v2 import (
    bench_karoubi_v2,
)
from quant_fund.models.quillen_ldev import (
    bench_quillen_ldev,
)
from quant_fund.models.seg_street import (
    bench_seg_street,
)
from quant_fund.models.vorst_descent import (
    bench_vorst_descent,
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


def bench_grayson_s_family(
    seed: int = _SEED + 3734,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "grayson_s",
            bench_grayson_s(seed),
        )
    )


def bench_karoubi_v2_family(
    seed: int = _SEED + 3735,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "karoubi_v2",
            bench_karoubi_v2(seed),
        )
    )


def bench_vorst_descent_family(
    seed: int = _SEED + 3736,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "vorst_descent",
            bench_vorst_descent(seed),
        )
    )


def bench_quillen_ldev_family(
    seed: int = _SEED + 3737,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "quillen_ldev",
            bench_quillen_ldev(seed),
        )
    )


def bench_fundamental_cat_family(
    seed: int = _SEED + 3738,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "fundamental_cat",
            bench_fundamental_cat(seed),
        )
    )


def bench_seg_street_family(
    seed: int = _SEED + 3739,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "seg_street",
            bench_seg_street(seed),
        )
    )
