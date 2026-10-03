"""Wave-544 Kleinian-groups bench adapters (SYNTHETIC-only)."""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.hyperbolic_3mfd import bench_hyperbolic_3mfd
from quant_fund.models.jorgensen_thurston import bench_jorgensen_thurston
from quant_fund.models.kleinian_group import bench_kleinian_group
from quant_fund.models.limit_set import bench_limit_set
from quant_fund.models.mostow_rigidity import bench_mostow_rigidity
from quant_fund.models.tameness_thm import bench_tameness_thm

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


def bench_kleinian_group_family(
    seed: int = _SEED + 3170,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "kleinian_group",
            bench_kleinian_group(seed),
        )
    )


def bench_limit_set_family(seed: int = _SEED + 3171) -> dict[str, float]:
    return _floats(_finite_blob("limit_set", bench_limit_set(seed)))


def bench_hyperbolic_3mfd_family(
    seed: int = _SEED + 3172,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "hyperbolic_3mfd",
            bench_hyperbolic_3mfd(seed),
        )
    )


def bench_mostow_rigidity_family(
    seed: int = _SEED + 3173,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "mostow_rigidity",
            bench_mostow_rigidity(seed),
        )
    )


def bench_jorgensen_thurston_family(
    seed: int = _SEED + 3174,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "jorgensen_thurston",
            bench_jorgensen_thurston(seed),
        )
    )


def bench_tameness_thm_family(seed: int = _SEED + 3175) -> dict[str, float]:
    return _floats(_finite_blob("tameness_thm", bench_tameness_thm(seed)))
