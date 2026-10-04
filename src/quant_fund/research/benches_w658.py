"""Wave-658 motivic-13 bench adapters (SYNTHETIC-only)."""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.beilinson_regulator import bench_beilinson_regulator
from quant_fund.models.f_motive import bench_f_motive
from quant_fund.models.hodge_motive import bench_hodge_motive
from quant_fund.models.motivic_galois import bench_motivic_galois
from quant_fund.models.period_realization import bench_period_realization
from quant_fund.models.tannakian_motive import bench_tannakian_motive

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


def bench_motivic_galois_family(
    seed: int = _SEED + 4700,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "motivic_galois",
            bench_motivic_galois(seed),
        )
    )


def bench_tannakian_motive_family(
    seed: int = _SEED + 4701,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "tannakian_motive",
            bench_tannakian_motive(seed),
        )
    )


def bench_period_realization_family(
    seed: int = _SEED + 4702,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "period_realization",
            bench_period_realization(seed),
        )
    )


def bench_beilinson_regulator_family(
    seed: int = _SEED + 4703,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "beilinson_regulator",
            bench_beilinson_regulator(seed),
        )
    )


def bench_hodge_motive_family(
    seed: int = _SEED + 4704,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "hodge_motive",
            bench_hodge_motive(seed),
        )
    )


def bench_f_motive_family(
    seed: int = _SEED + 4705,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "f_motive",
            bench_f_motive(seed),
        )
    )
