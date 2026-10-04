"""Wave-622 prismatic bench adapters (SYNTHETIC-only)."""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.delta_ring import bench_delta_ring
from quant_fund.models.hodge_tate import bench_hodge_tate
from quant_fund.models.nygaard2 import bench_nygaard2
from quant_fund.models.prism2 import bench_prism2
from quant_fund.models.prismatic_crystal import (
    bench_prismatic_crystal,
)
from quant_fund.models.prismatic_site import (
    bench_prismatic_site,
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


def bench_prism2_family(seed: int = _SEED + 3638) -> dict[str, float]:
    return _floats(_finite_blob("prism2", bench_prism2(seed)))


def bench_prismatic_site_family(
    seed: int = _SEED + 3639,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "prismatic_site",
            bench_prismatic_site(seed),
        )
    )


def bench_delta_ring_family(
    seed: int = _SEED + 3640,
) -> dict[str, float]:
    return _floats(_finite_blob("delta_ring", bench_delta_ring(seed)))


def bench_prismatic_crystal_family(
    seed: int = _SEED + 3641,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "prismatic_crystal",
            bench_prismatic_crystal(seed),
        )
    )


def bench_hodge_tate_family(
    seed: int = _SEED + 3642,
) -> dict[str, float]:
    return _floats(_finite_blob("hodge_tate", bench_hodge_tate(seed)))


def bench_nygaard2_family(
    seed: int = _SEED + 3643,
) -> dict[str, float]:
    return _floats(_finite_blob("nygaard2", bench_nygaard2(seed)))
