"""Wave-723 Iwasawa/Euler-system bench adapters (SYNTHETIC-only)."""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.euler_system import bench_euler_system
from quant_fund.models.gross_zagier import bench_gross_zagier
from quant_fund.models.iwasawa_motive import bench_iwasawa_motive
from quant_fund.models.kolyvagin_sys import bench_kolyvagin_sys
from quant_fund.models.perrin_riou import bench_perrin_riou
from quant_fund.models.rubin_main_conj import (
    bench_rubin_main_conj,
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


def bench_gross_zagier_family(
    seed: int = _SEED + 11200,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "gross_zagier",
            bench_gross_zagier(seed),
        )
    )


def bench_kolyvagin_sys_family(
    seed: int = _SEED + 11201,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "kolyvagin_sys",
            bench_kolyvagin_sys(seed),
        )
    )


def bench_euler_system_family(
    seed: int = _SEED + 11202,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "euler_system",
            bench_euler_system(seed),
        )
    )


def bench_iwasawa_motive_family(
    seed: int = _SEED + 11203,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "iwasawa_motive",
            bench_iwasawa_motive(seed),
        )
    )


def bench_rubin_main_conj_family(
    seed: int = _SEED + 11204,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "rubin_main_conj",
            bench_rubin_main_conj(seed),
        )
    )


def bench_perrin_riou_family(
    seed: int = _SEED + 11205,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "perrin_riou",
            bench_perrin_riou(seed),
        )
    )
