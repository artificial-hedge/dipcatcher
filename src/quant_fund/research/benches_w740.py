"""Wave-740 percolation bench adapters (SYNTHETIC-only)."""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.cardy_formula import bench_cardy_formula
from quant_fund.models.duminil_copin import bench_duminil_copin
from quant_fund.models.grimmett_percolation import (
    bench_grimmett_percolation,
)
from quant_fund.models.kesten_percolation import (
    bench_kesten_percolation,
)
from quant_fund.models.russo_seymour import bench_russo_seymour
from quant_fund.models.smirnov_percolation import (
    bench_smirnov_percolation,
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


def bench_smirnov_percolation_family(
    seed: int = _SEED + 12900,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "smirnov_percolation",
            bench_smirnov_percolation(seed),
        )
    )


def bench_duminil_copin_family(
    seed: int = _SEED + 12901,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "duminil_copin",
            bench_duminil_copin(seed),
        )
    )


def bench_kesten_percolation_family(
    seed: int = _SEED + 12902,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "kesten_percolation",
            bench_kesten_percolation(seed),
        )
    )


def bench_cardy_formula_family(
    seed: int = _SEED + 12903,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "cardy_formula",
            bench_cardy_formula(seed),
        )
    )


def bench_russo_seymour_family(
    seed: int = _SEED + 12904,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "russo_seymour",
            bench_russo_seymour(seed),
        )
    )


def bench_grimmett_percolation_family(
    seed: int = _SEED + 12905,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "grimmett_percolation",
            bench_grimmett_percolation(seed),
        )
    )
