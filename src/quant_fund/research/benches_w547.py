"""Wave-547 4-manifold bench adapters (SYNTHETIC-only)."""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.donaldson_thm import bench_donaldson_thm
from quant_fund.models.exotic_r4 import bench_exotic_r4
from quant_fund.models.four_mfd import bench_four_mfd
from quant_fund.models.freedman_thm import bench_freedman_thm
from quant_fund.models.intersection_form import bench_intersection_form
from quant_fund.models.seiberg_witten import bench_seiberg_witten

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


def bench_four_mfd_family(seed: int = _SEED + 3188) -> dict[str, float]:
    return _floats(_finite_blob("four_mfd", bench_four_mfd(seed)))


def bench_donaldson_thm_family(seed: int = _SEED + 3189) -> dict[str, float]:
    return _floats(_finite_blob("donaldson_thm", bench_donaldson_thm(seed)))


def bench_seiberg_witten_family(
    seed: int = _SEED + 3190,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "seiberg_witten",
            bench_seiberg_witten(seed),
        )
    )


def bench_exotic_r4_family(seed: int = _SEED + 3191) -> dict[str, float]:
    return _floats(_finite_blob("exotic_r4", bench_exotic_r4(seed)))


def bench_intersection_form_family(
    seed: int = _SEED + 3192,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "intersection_form",
            bench_intersection_form(seed),
        )
    )


def bench_freedman_thm_family(seed: int = _SEED + 3193) -> dict[str, float]:
    return _floats(_finite_blob("freedman_thm", bench_freedman_thm(seed)))
