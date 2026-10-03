"""Wave-414 homotopy-6 bench adapters (SYNTHETIC-only)."""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.adams_ss import bench_adams_ss
from quant_fund.models.cofiber import bench_cofiber
from quant_fund.models.exact_couple import bench_exact_couple
from quant_fund.models.obstruction import bench_obstruction
from quant_fund.models.stable_homotopy import bench_stable_homotopy
from quant_fund.models.whitehead_thm import bench_whitehead_thm

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


def bench_exact_couple_family(
    seed: int = _SEED + 2390,
) -> dict[str, float]:
    return _floats(_finite_blob("exact_couple", bench_exact_couple(seed)))


def bench_adams_ss_family(seed: int = _SEED + 2391) -> dict[str, float]:
    return _floats(_finite_blob("adams_ss", bench_adams_ss(seed)))


def bench_stable_homotopy_family(
    seed: int = _SEED + 2392,
) -> dict[str, float]:
    return _floats(_finite_blob("stable_homotopy", bench_stable_homotopy(seed)))


def bench_whitehead_thm_family(
    seed: int = _SEED + 2393,
) -> dict[str, float]:
    return _floats(_finite_blob("whitehead_thm", bench_whitehead_thm(seed)))


def bench_obstruction_family(
    seed: int = _SEED + 2394,
) -> dict[str, float]:
    return _floats(_finite_blob("obstruction", bench_obstruction(seed)))


def bench_cofiber_family(seed: int = _SEED + 2395) -> dict[str, float]:
    return _floats(_finite_blob("cofiber", bench_cofiber(seed)))
