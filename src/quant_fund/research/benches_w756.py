"""Wave-756 GFF bench adapters (SYNTHETIC-only)."""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.berestycki_gff import bench_berestycki_gff
from quant_fund.models.duplantier_sheffield import (
    bench_duplantier_sheffield,
)
from quant_fund.models.houchmandzadeh_gff import (
    bench_houchmandzadeh_gff,
)
from quant_fund.models.nick_gff import bench_nick_gff
from quant_fund.models.sheffield_miller import (
    bench_sheffield_miller,
)
from quant_fund.models.wiegmann_zabrodin import (
    bench_wiegmann_zabrodin,
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


def bench_berestycki_gff_family(
    seed: int = _SEED + 14500,
) -> dict[str, float]:
    return _floats(_finite_blob("berestycki_gff", bench_berestycki_gff(seed)))


def bench_duplantier_sheffield_family(
    seed: int = _SEED + 14501,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "duplantier_sheffield",
            bench_duplantier_sheffield(seed),
        )
    )


def bench_houchmandzadeh_gff_family(
    seed: int = _SEED + 14502,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "houchmandzadeh_gff",
            bench_houchmandzadeh_gff(seed),
        )
    )


def bench_nick_gff_family(
    seed: int = _SEED + 14503,
) -> dict[str, float]:
    return _floats(_finite_blob("nick_gff", bench_nick_gff(seed)))


def bench_sheffield_miller_family(
    seed: int = _SEED + 14504,
) -> dict[str, float]:
    return _floats(_finite_blob("sheffield_miller", bench_sheffield_miller(seed)))


def bench_wiegmann_zabrodin_family(
    seed: int = _SEED + 14505,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "wiegmann_zabrodin",
            bench_wiegmann_zabrodin(seed),
        )
    )
