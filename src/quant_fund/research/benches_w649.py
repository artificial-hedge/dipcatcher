"""Wave-649 prismatic-3 bench adapters (SYNTHETIC-only)."""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.breuil_prism import bench_breuil_prism
from quant_fund.models.cartier_prism import bench_cartier_prism
from quant_fund.models.filtered_prism import bench_filtered_prism
from quant_fund.models.frobenius_prism import bench_frobenius_prism
from quant_fund.models.prism_site2 import bench_prism_site2
from quant_fund.models.stacky_prism import bench_stacky_prism

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


def bench_prism_site2_family(
    seed: int = _SEED + 3800,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "prism_site2",
            bench_prism_site2(seed),
        )
    )


def bench_cartier_prism_family(
    seed: int = _SEED + 3801,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "cartier_prism",
            bench_cartier_prism(seed),
        )
    )


def bench_breuil_prism_family(
    seed: int = _SEED + 3802,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "breuil_prism",
            bench_breuil_prism(seed),
        )
    )


def bench_filtered_prism_family(
    seed: int = _SEED + 3803,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "filtered_prism",
            bench_filtered_prism(seed),
        )
    )


def bench_frobenius_prism_family(
    seed: int = _SEED + 3804,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "frobenius_prism",
            bench_frobenius_prism(seed),
        )
    )


def bench_stacky_prism_family(
    seed: int = _SEED + 3805,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "stacky_prism",
            bench_stacky_prism(seed),
        )
    )
