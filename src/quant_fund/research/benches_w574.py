"""Wave-574 differential-topology-2 bench adapters (SYNTHETIC-only)."""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.exotic_sphere import bench_exotic_sphere
from quant_fund.models.immersion_thm import bench_immersion_thm
from quant_fund.models.kervaire_milnor import (
    bench_kervaire_milnor,
)
from quant_fund.models.smale_hcob import bench_smale_hcob
from quant_fund.models.surgery_theory import bench_surgery_theory
from quant_fund.models.whitney_trick import bench_whitney_trick

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


def bench_exotic_sphere_family(seed: int = _SEED + 3350) -> dict[str, float]:
    return _floats(_finite_blob("exotic_sphere", bench_exotic_sphere(seed)))


def bench_kervaire_milnor_family(
    seed: int = _SEED + 3351,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "kervaire_milnor",
            bench_kervaire_milnor(seed),
        )
    )


def bench_surgery_theory_family(
    seed: int = _SEED + 3352,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "surgery_theory",
            bench_surgery_theory(seed),
        )
    )


def bench_smale_hcob_family(seed: int = _SEED + 3353) -> dict[str, float]:
    return _floats(_finite_blob("smale_hcob", bench_smale_hcob(seed)))


def bench_whitney_trick_family(seed: int = _SEED + 3354) -> dict[str, float]:
    return _floats(_finite_blob("whitney_trick", bench_whitney_trick(seed)))


def bench_immersion_thm_family(seed: int = _SEED + 3355) -> dict[str, float]:
    return _floats(_finite_blob("immersion_thm", bench_immersion_thm(seed)))
