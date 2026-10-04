"""Wave-556 Teichmueller-theory bench adapters (SYNTHETIC-only)."""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.earthquake_map import bench_earthquake_map
from quant_fund.models.extremal_length import bench_extremal_length
from quant_fund.models.mapping_class import bench_mapping_class
from quant_fund.models.pseudo_anosov import bench_pseudo_anosov
from quant_fund.models.quadratic_diff import bench_quadratic_diff
from quant_fund.models.weil_petersson import bench_weil_petersson

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


def bench_weil_petersson_family(seed: int = _SEED + 3242) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "weil_petersson",
            bench_weil_petersson(seed),
        )
    )


def bench_mapping_class_family(seed: int = _SEED + 3243) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "mapping_class",
            bench_mapping_class(seed),
        )
    )


def bench_quadratic_diff_family(seed: int = _SEED + 3244) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "quadratic_diff",
            bench_quadratic_diff(seed),
        )
    )


def bench_earthquake_map_family(seed: int = _SEED + 3245) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "earthquake_map",
            bench_earthquake_map(seed),
        )
    )


def bench_extremal_length_family(
    seed: int = _SEED + 3246,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "extremal_length",
            bench_extremal_length(seed),
        )
    )


def bench_pseudo_anosov_family(seed: int = _SEED + 3247) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "pseudo_anosov",
            bench_pseudo_anosov(seed),
        )
    )
