"""Wave-704 higher-algebra-10 bench adapters (SYNTHETIC-only)."""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.braces_e5 import bench_braces_e5
from quant_fund.models.delooping3 import bench_delooping3
from quant_fund.models.higher_algebra9 import (
    bench_higher_algebra9,
)
from quant_fund.models.koszul_duality3 import (
    bench_koszul_duality3,
)
from quant_fund.models.operad_infty5 import bench_operad_infty5
from quant_fund.models.operad_swiss4 import bench_operad_swiss4

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


def bench_higher_algebra9_family(
    seed: int = _SEED + 9300,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "higher_algebra9",
            bench_higher_algebra9(seed),
        )
    )


def bench_operad_infty5_family(
    seed: int = _SEED + 9301,
) -> dict[str, float]:
    return _floats(_finite_blob("operad_infty5", bench_operad_infty5(seed)))


def bench_operad_swiss4_family(
    seed: int = _SEED + 9302,
) -> dict[str, float]:
    return _floats(_finite_blob("operad_swiss4", bench_operad_swiss4(seed)))


def bench_koszul_duality3_family(
    seed: int = _SEED + 9303,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "koszul_duality3",
            bench_koszul_duality3(seed),
        )
    )


def bench_braces_e5_family(
    seed: int = _SEED + 9304,
) -> dict[str, float]:
    return _floats(_finite_blob("braces_e5", bench_braces_e5(seed)))


def bench_delooping3_family(
    seed: int = _SEED + 9305,
) -> dict[str, float]:
    return _floats(_finite_blob("delooping3", bench_delooping3(seed)))
