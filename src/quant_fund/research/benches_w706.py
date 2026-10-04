"""Wave-706 chromatic-9 bench adapters (SYNTHETIC-only)."""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.chromatic_base import bench_chromatic_base
from quant_fund.models.chromatic_layer import (
    bench_chromatic_layer,
)
from quant_fund.models.chromatic_square2 import (
    bench_chromatic_square2,
)
from quant_fund.models.elliptic_morava import (
    bench_elliptic_morava,
)
from quant_fund.models.lubin_tate3 import bench_lubin_tate3
from quant_fund.models.morava_maven import bench_morava_maven

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


def bench_chromatic_layer_family(
    seed: int = _SEED + 9500,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "chromatic_layer",
            bench_chromatic_layer(seed),
        )
    )


def bench_morava_maven_family(
    seed: int = _SEED + 9501,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "morava_maven",
            bench_morava_maven(seed),
        )
    )


def bench_chromatic_square2_family(
    seed: int = _SEED + 9502,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "chromatic_square2",
            bench_chromatic_square2(seed),
        )
    )


def bench_lubin_tate3_family(
    seed: int = _SEED + 9503,
) -> dict[str, float]:
    return _floats(_finite_blob("lubin_tate3", bench_lubin_tate3(seed)))


def bench_elliptic_morava_family(
    seed: int = _SEED + 9504,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "elliptic_morava",
            bench_elliptic_morava(seed),
        )
    )


def bench_chromatic_base_family(
    seed: int = _SEED + 9505,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "chromatic_base",
            bench_chromatic_base(seed),
        )
    )
