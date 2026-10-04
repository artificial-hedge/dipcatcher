"""Wave-736 LQG bench adapters (SYNTHETIC-only)."""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.aru_powell import bench_aru_powell
from quant_fund.models.berestycki_sheffield import (
    bench_berestycki_sheffield,
)
from quant_fund.models.bisbisot_sheffield import (
    bench_bisbisot_sheffield,
)
from quant_fund.models.dhms_lqg import bench_dhms_lqg
from quant_fund.models.huang_rhodes import bench_huang_rhodes
from quant_fund.models.sheffield_gff import bench_sheffield_gff

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


def bench_sheffield_gff_family(
    seed: int = _SEED + 12500,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "sheffield_gff",
            bench_sheffield_gff(seed),
        )
    )


def bench_berestycki_sheffield_family(
    seed: int = _SEED + 12501,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "berestycki_sheffield",
            bench_berestycki_sheffield(seed),
        )
    )


def bench_aru_powell_family(
    seed: int = _SEED + 12502,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "aru_powell",
            bench_aru_powell(seed),
        )
    )


def bench_huang_rhodes_family(
    seed: int = _SEED + 12503,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "huang_rhodes",
            bench_huang_rhodes(seed),
        )
    )


def bench_bisbisot_sheffield_family(
    seed: int = _SEED + 12504,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "bisbisot_sheffield",
            bench_bisbisot_sheffield(seed),
        )
    )


def bench_dhms_lqg_family(
    seed: int = _SEED + 12505,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "dhms_lqg",
            bench_dhms_lqg(seed),
        )
    )
