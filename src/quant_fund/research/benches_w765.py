"""Wave-765 LIL/LLN bench adapters (SYNTHETIC-only)."""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.chung_lil import bench_chung_lil
from quant_fund.models.glivenko_cantelli import (
    bench_glivenko_cantelli,
)
from quant_fund.models.khintchine_lln import bench_khintchine_lln
from quant_fund.models.kolmogorov_3series import (
    bench_kolmogorov_3series,
)
from quant_fund.models.levy_convergence import (
    bench_levy_convergence,
)
from quant_fund.models.strassen_lil import bench_strassen_lil

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


def bench_strassen_lil_family(
    seed: int = _SEED + 15400,
) -> dict[str, float]:
    return _floats(_finite_blob("strassen_lil", bench_strassen_lil(seed)))


def bench_chung_lil_family(
    seed: int = _SEED + 15401,
) -> dict[str, float]:
    return _floats(_finite_blob("chung_lil", bench_chung_lil(seed)))


def bench_kolmogorov_3series_family(
    seed: int = _SEED + 15402,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "kolmogorov_3series",
            bench_kolmogorov_3series(seed),
        )
    )


def bench_khintchine_lln_family(
    seed: int = _SEED + 15403,
) -> dict[str, float]:
    return _floats(_finite_blob("khintchine_lln", bench_khintchine_lln(seed)))


def bench_levy_convergence_family(
    seed: int = _SEED + 15404,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "levy_convergence",
            bench_levy_convergence(seed),
        )
    )


def bench_glivenko_cantelli_family(
    seed: int = _SEED + 15405,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "glivenko_cantelli",
            bench_glivenko_cantelli(seed),
        )
    )
