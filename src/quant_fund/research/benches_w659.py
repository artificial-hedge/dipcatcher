"""Wave-659 motivic-14 bench adapters (SYNTHETIC-only)."""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.absolute_cohom import bench_absolute_cohom
from quant_fund.models.motivic_pairing import bench_motivic_pairing
from quant_fund.models.motivic_tate2 import bench_motivic_tate2
from quant_fund.models.motivic_weight import bench_motivic_weight
from quant_fund.models.norimotive2 import bench_norimotive2
from quant_fund.models.tate_triple import bench_tate_triple

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


def bench_norimotive2_family(
    seed: int = _SEED + 4800,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "norimotive2",
            bench_norimotive2(seed),
        )
    )


def bench_motivic_tate2_family(
    seed: int = _SEED + 4801,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "motivic_tate2",
            bench_motivic_tate2(seed),
        )
    )


def bench_absolute_cohom_family(
    seed: int = _SEED + 4802,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "absolute_cohom",
            bench_absolute_cohom(seed),
        )
    )


def bench_motivic_weight_family(
    seed: int = _SEED + 4803,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "motivic_weight",
            bench_motivic_weight(seed),
        )
    )


def bench_tate_triple_family(
    seed: int = _SEED + 4804,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "tate_triple",
            bench_tate_triple(seed),
        )
    )


def bench_motivic_pairing_family(
    seed: int = _SEED + 4805,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "motivic_pairing",
            bench_motivic_pairing(seed),
        )
    )
