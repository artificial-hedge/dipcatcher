"""Wave-782 filtration/Jacod-Shiryaev bench adapters (SYNTHETIC-only)."""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.ffusion_lims import bench_ffusion_lims
from quant_fund.models.filt_proc import bench_filt_proc
from quant_fund.models.jacod_shiryaev import (
    bench_jacod_shiryaev,
)
from quant_fund.models.kunita_watanabe import (
    bench_kunita_watanabe,
)
from quant_fund.models.pinsky_proc import bench_pinsky_proc
from quant_fund.models.slivnyak import bench_slivnyak

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


def bench_pinsky_proc_family(
    seed: int = _SEED + 17100,
) -> dict[str, float]:
    return _floats(_finite_blob("pinsky_proc", bench_pinsky_proc(seed)))


def bench_ffusion_lims_family(
    seed: int = _SEED + 17101,
) -> dict[str, float]:
    return _floats(_finite_blob("ffusion_lims", bench_ffusion_lims(seed)))


def bench_kunita_watanabe_family(
    seed: int = _SEED + 17102,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "kunita_watanabe",
            bench_kunita_watanabe(seed),
        )
    )


def bench_filt_proc_family(
    seed: int = _SEED + 17103,
) -> dict[str, float]:
    return _floats(_finite_blob("filt_proc", bench_filt_proc(seed)))


def bench_slivnyak_family(
    seed: int = _SEED + 17104,
) -> dict[str, float]:
    return _floats(_finite_blob("slivnyak", bench_slivnyak(seed)))


def bench_jacod_shiryaev_family(
    seed: int = _SEED + 17105,
) -> dict[str, float]:
    return _floats(_finite_blob("jacod_shiryaev", bench_jacod_shiryaev(seed)))
