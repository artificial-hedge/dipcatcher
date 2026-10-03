"""Wave-726 Galois-deformation bench adapters (SYNTHETIC-only)."""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.diamond_taylor_wiles import (
    bench_diamond_taylor_wiles,
)
from quant_fund.models.jetchev_skinner import (
    bench_jetchev_skinner,
)
from quant_fund.models.kisin_crystalline import (
    bench_kisin_crystalline,
)
from quant_fund.models.mazur_deform import bench_mazur_deform
from quant_fund.models.wan_sss import bench_wan_sss
from quant_fund.models.wiles_taylor import bench_wiles_taylor

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


def bench_jetchev_skinner_family(
    seed: int = _SEED + 11500,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "jetchev_skinner",
            bench_jetchev_skinner(seed),
        )
    )


def bench_wan_sss_family(
    seed: int = _SEED + 11501,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "wan_sss",
            bench_wan_sss(seed),
        )
    )


def bench_wiles_taylor_family(
    seed: int = _SEED + 11502,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "wiles_taylor",
            bench_wiles_taylor(seed),
        )
    )


def bench_diamond_taylor_wiles_family(
    seed: int = _SEED + 11503,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "diamond_taylor_wiles",
            bench_diamond_taylor_wiles(seed),
        )
    )


def bench_kisin_crystalline_family(
    seed: int = _SEED + 11504,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "kisin_crystalline",
            bench_kisin_crystalline(seed),
        )
    )


def bench_mazur_deform_family(
    seed: int = _SEED + 11505,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "mazur_deform",
            bench_mazur_deform(seed),
        )
    )
