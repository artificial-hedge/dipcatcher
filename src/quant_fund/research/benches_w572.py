"""Wave-572 abelian-varieties bench adapters (SYNTHETIC-only)."""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.abelian_variety import bench_abelian_variety
from quant_fund.models.faltings_thm import bench_faltings_thm
from quant_fund.models.isogeny_av import bench_isogeny_av
from quant_fund.models.mordell_weil_av import bench_mordell_weil_av
from quant_fund.models.shafarevich_conj import (
    bench_shafarevich_conj,
)
from quant_fund.models.tate_module import bench_tate_module

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


def bench_abelian_variety_family(
    seed: int = _SEED + 3338,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "abelian_variety",
            bench_abelian_variety(seed),
        )
    )


def bench_isogeny_av_family(seed: int = _SEED + 3339) -> dict[str, float]:
    return _floats(_finite_blob("isogeny_av", bench_isogeny_av(seed)))


def bench_tate_module_family(seed: int = _SEED + 3340) -> dict[str, float]:
    return _floats(_finite_blob("tate_module", bench_tate_module(seed)))


def bench_shafarevich_conj_family(
    seed: int = _SEED + 3341,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "shafarevich_conj",
            bench_shafarevich_conj(seed),
        )
    )


def bench_faltings_thm_family(seed: int = _SEED + 3342) -> dict[str, float]:
    return _floats(_finite_blob("faltings_thm", bench_faltings_thm(seed)))


def bench_mordell_weil_av_family(
    seed: int = _SEED + 3343,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "mordell_weil_av",
            bench_mordell_weil_av(seed),
        )
    )
