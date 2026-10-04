"""Wave-766 empirical-process-2 bench adapters (SYNTHETIC-only)."""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.bounded_lip import bench_bounded_lip
from quant_fund.models.bracketing_ent import bench_bracketing_ent
from quant_fund.models.dudley_theorem import bench_dudley_theorem
from quant_fund.models.dvoretzky_thm import bench_dvoretzky_thm
from quant_fund.models.varadarajan_thm import (
    bench_varadarajan_thm,
)
from quant_fund.models.vc_class import bench_vc_class

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


def bench_dudley_theorem_family(
    seed: int = _SEED + 15500,
) -> dict[str, float]:
    return _floats(_finite_blob("dudley_theorem", bench_dudley_theorem(seed)))


def bench_varadarajan_thm_family(
    seed: int = _SEED + 15501,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "varadarajan_thm",
            bench_varadarajan_thm(seed),
        )
    )


def bench_dvoretzky_thm_family(
    seed: int = _SEED + 15502,
) -> dict[str, float]:
    return _floats(_finite_blob("dvoretzky_thm", bench_dvoretzky_thm(seed)))


def bench_vc_class_family(
    seed: int = _SEED + 15503,
) -> dict[str, float]:
    return _floats(_finite_blob("vc_class", bench_vc_class(seed)))


def bench_bracketing_ent_family(
    seed: int = _SEED + 15504,
) -> dict[str, float]:
    return _floats(_finite_blob("bracketing_ent", bench_bracketing_ent(seed)))


def bench_bounded_lip_family(
    seed: int = _SEED + 15505,
) -> dict[str, float]:
    return _floats(_finite_blob("bounded_lip", bench_bounded_lip(seed)))
