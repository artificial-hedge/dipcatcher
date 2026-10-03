"""Wave-735 SLE-2 bench adapters (SYNTHETIC-only)."""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.beffara_sle import bench_beffara_sle
from quant_fund.models.benoist_sle import bench_benoist_sle
from quant_fund.models.holden_sle import bench_holden_sle
from quant_fund.models.kemppainen_sle import (
    bench_kemppainen_sle,
)
from quant_fund.models.viklund_sle import bench_viklund_sle
from quant_fund.models.zykin_sle import bench_zykin_sle

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


def bench_beffara_sle_family(
    seed: int = _SEED + 12400,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "beffara_sle",
            bench_beffara_sle(seed),
        )
    )


def bench_kemppainen_sle_family(
    seed: int = _SEED + 12401,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "kemppainen_sle",
            bench_kemppainen_sle(seed),
        )
    )


def bench_zykin_sle_family(
    seed: int = _SEED + 12402,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "zykin_sle",
            bench_zykin_sle(seed),
        )
    )


def bench_viklund_sle_family(
    seed: int = _SEED + 12403,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "viklund_sle",
            bench_viklund_sle(seed),
        )
    )


def bench_benoist_sle_family(
    seed: int = _SEED + 12404,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "benoist_sle",
            bench_benoist_sle(seed),
        )
    )


def bench_holden_sle_family(
    seed: int = _SEED + 12405,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "holden_sle",
            bench_holden_sle(seed),
        )
    )
