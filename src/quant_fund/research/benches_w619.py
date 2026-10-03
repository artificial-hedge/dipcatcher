"""Wave-619 homotopy-14 bench adapters (SYNTHETIC-only)."""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.andersen_lannes import (
    bench_andersen_lannes,
)
from quant_fund.models.chromatic_hopkins import (
    bench_chromatic_hopkins,
)
from quant_fund.models.devissage_ss import bench_devissage_ss
from quant_fund.models.tame_htpy import bench_tame_htpy
from quant_fund.models.thick_spectrum import (
    bench_thick_spectrum,
)
from quant_fund.models.unstable_htpy import bench_unstable_htpy

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


def bench_unstable_htpy_family(
    seed: int = _SEED + 3620,
) -> dict[str, float]:
    return _floats(_finite_blob("unstable_htpy", bench_unstable_htpy(seed)))


def bench_tame_htpy_family(
    seed: int = _SEED + 3621,
) -> dict[str, float]:
    return _floats(_finite_blob("tame_htpy", bench_tame_htpy(seed)))


def bench_devissage_ss_family(
    seed: int = _SEED + 3622,
) -> dict[str, float]:
    return _floats(_finite_blob("devissage_ss", bench_devissage_ss(seed)))


def bench_andersen_lannes_family(
    seed: int = _SEED + 3623,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "andersen_lannes",
            bench_andersen_lannes(seed),
        )
    )


def bench_chromatic_hopkins_family(
    seed: int = _SEED + 3624,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "chromatic_hopkins",
            bench_chromatic_hopkins(seed),
        )
    )


def bench_thick_spectrum_family(
    seed: int = _SEED + 3625,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "thick_spectrum",
            bench_thick_spectrum(seed),
        )
    )
