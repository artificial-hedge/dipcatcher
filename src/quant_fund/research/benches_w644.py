"""Wave-644 algebraic-K-8 bench adapters (SYNTHETIC-only)."""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.allday_k import bench_allday_k
from quant_fund.models.hall_alg import bench_hall_alg
from quant_fund.models.residue_k import bench_residue_k
from quant_fund.models.s_multicat import bench_s_multicat
from quant_fund.models.suslin_wagoner import (
    bench_suslin_wagoner,
)
from quant_fund.models.weibel_nil import bench_weibel_nil

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


def bench_s_multicat_family(
    seed: int = _SEED + 3770,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "s_multicat",
            bench_s_multicat(seed),
        )
    )


def bench_allday_k_family(
    seed: int = _SEED + 3771,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "allday_k",
            bench_allday_k(seed),
        )
    )


def bench_residue_k_family(
    seed: int = _SEED + 3772,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "residue_k",
            bench_residue_k(seed),
        )
    )


def bench_suslin_wagoner_family(
    seed: int = _SEED + 3773,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "suslin_wagoner",
            bench_suslin_wagoner(seed),
        )
    )


def bench_weibel_nil_family(
    seed: int = _SEED + 3774,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "weibel_nil",
            bench_weibel_nil(seed),
        )
    )


def bench_hall_alg_family(
    seed: int = _SEED + 3775,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "hall_alg",
            bench_hall_alg(seed),
        )
    )
