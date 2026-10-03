"""Wave-608 sheaf-4 bench adapters (SYNTHETIC-only)."""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.etale_descent import (
    bench_etale_descent,
)
from quant_fund.models.etale_morphism import (
    bench_etale_morphism,
)
from quant_fund.models.fppf_site import bench_fppf_site
from quant_fund.models.fpqc_site import bench_fpqc_site
from quant_fund.models.ladic_sheaf import bench_ladic_sheaf
from quant_fund.models.lisse_sheaf import bench_lisse_sheaf

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


def bench_etale_descent_family(
    seed: int = _SEED + 3554,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "etale_descent",
            bench_etale_descent(seed),
        )
    )


def bench_etale_morphism_family(
    seed: int = _SEED + 3555,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "etale_morphism",
            bench_etale_morphism(seed),
        )
    )


def bench_fppf_site_family(seed: int = _SEED + 3556) -> dict[str, float]:
    return _floats(_finite_blob("fppf_site", bench_fppf_site(seed)))


def bench_fpqc_site_family(seed: int = _SEED + 3557) -> dict[str, float]:
    return _floats(_finite_blob("fpqc_site", bench_fpqc_site(seed)))


def bench_ladic_sheaf_family(
    seed: int = _SEED + 3558,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "ladic_sheaf",
            bench_ladic_sheaf(seed),
        )
    )


def bench_lisse_sheaf_family(
    seed: int = _SEED + 3559,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "lisse_sheaf",
            bench_lisse_sheaf(seed),
        )
    )
