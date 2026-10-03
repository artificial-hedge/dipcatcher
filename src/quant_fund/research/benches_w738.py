"""Wave-738 Brownian-map bench adapters (SYNTHETIC-only)."""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.abraham_bipartite import (
    bench_abraham_bipartite,
)
from quant_fund.models.bettinelli_jacob import (
    bench_bettinelli_jacob,
)
from quant_fund.models.chapuy_dolega import bench_chapuy_dolega
from quant_fund.models.curien_legall import bench_curien_legall
from quant_fund.models.le_gall_miermont import (
    bench_le_gall_miermont,
)
from quant_fund.models.marckert_mokkadem import (
    bench_marckert_mokkadem,
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


def bench_marckert_mokkadem_family(
    seed: int = _SEED + 12700,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "marckert_mokkadem",
            bench_marckert_mokkadem(seed),
        )
    )


def bench_le_gall_miermont_family(
    seed: int = _SEED + 12701,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "le_gall_miermont",
            bench_le_gall_miermont(seed),
        )
    )


def bench_curien_legall_family(
    seed: int = _SEED + 12702,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "curien_legall",
            bench_curien_legall(seed),
        )
    )


def bench_abraham_bipartite_family(
    seed: int = _SEED + 12703,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "abraham_bipartite",
            bench_abraham_bipartite(seed),
        )
    )


def bench_bettinelli_jacob_family(
    seed: int = _SEED + 12704,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "bettinelli_jacob",
            bench_bettinelli_jacob(seed),
        )
    )


def bench_chapuy_dolega_family(
    seed: int = _SEED + 12705,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "chapuy_dolega",
            bench_chapuy_dolega(seed),
        )
    )
