"""Wave-692 higher-algebra-9 bench adapters (SYNTHETIC-only)."""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.centralizer_alg2 import (
    bench_centralizer_alg2,
)
from quant_fund.models.e5_algebra import bench_e5_algebra
from quant_fund.models.factorization_hom3 import (
    bench_factorization_hom3,
)
from quant_fund.models.framed_discs import (
    bench_framed_discs,
)
from quant_fund.models.little_cubes2 import (
    bench_little_cubes2,
)
from quant_fund.models.swiss_cheese3 import (
    bench_swiss_cheese3,
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


def bench_e5_algebra_family(
    seed: int = _SEED + 8100,
) -> dict[str, float]:
    return _floats(_finite_blob("e5_algebra", bench_e5_algebra(seed)))


def bench_little_cubes2_family(
    seed: int = _SEED + 8101,
) -> dict[str, float]:
    return _floats(_finite_blob("little_cubes2", bench_little_cubes2(seed)))


def bench_swiss_cheese3_family(
    seed: int = _SEED + 8102,
) -> dict[str, float]:
    return _floats(_finite_blob("swiss_cheese3", bench_swiss_cheese3(seed)))


def bench_framed_discs_family(
    seed: int = _SEED + 8103,
) -> dict[str, float]:
    return _floats(_finite_blob("framed_discs", bench_framed_discs(seed)))


def bench_factorization_hom3_family(
    seed: int = _SEED + 8104,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "factorization_hom3",
            bench_factorization_hom3(seed),
        )
    )


def bench_centralizer_alg2_family(
    seed: int = _SEED + 8105,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "centralizer_alg2",
            bench_centralizer_alg2(seed),
        )
    )
