"""Wave-567 algebraic-combinatorics bench adapters (SYNTHETIC-only)."""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.berenstein_zelevinsky import (
    bench_berenstein_zelevinsky,
)
from quant_fund.models.honeycomb_tiling import bench_honeycomb_tiling
from quant_fund.models.knuth_rsk import bench_knuth_rsk
from quant_fund.models.littlewood_richardson import (
    bench_littlewood_richardson,
)
from quant_fund.models.macdonald_poly import bench_macdonald_poly
from quant_fund.models.schubert_calc import bench_schubert_calc

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


def bench_littlewood_richardson_family(
    seed: int = _SEED + 3308,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "littlewood_richardson",
            bench_littlewood_richardson(seed),
        )
    )


def bench_knuth_rsk_family(seed: int = _SEED + 3309) -> dict[str, float]:
    return _floats(_finite_blob("knuth_rsk", bench_knuth_rsk(seed)))


def bench_macdonald_poly_family(
    seed: int = _SEED + 3310,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "macdonald_poly",
            bench_macdonald_poly(seed),
        )
    )


def bench_schubert_calc_family(seed: int = _SEED + 3311) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "schubert_calc",
            bench_schubert_calc(seed),
        )
    )


def bench_honeycomb_tiling_family(
    seed: int = _SEED + 3312,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "honeycomb_tiling",
            bench_honeycomb_tiling(seed),
        )
    )


def bench_berenstein_zelevinsky_family(
    seed: int = _SEED + 3313,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "berenstein_zelevinsky",
            bench_berenstein_zelevinsky(seed),
        )
    )
