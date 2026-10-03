"""Wave-584 condensed-3 bench adapters (SYNTHETIC-only)."""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.analytic_sheaf import (
    bench_analytic_sheaf,
)
from quant_fund.models.clausen_scholze2 import (
    bench_clausen_scholze2,
)
from quant_fund.models.nuclear_space import (
    bench_nuclear_space,
)
from quant_fund.models.proetale_site2 import (
    bench_proetale_site2,
)
from quant_fund.models.solid_cohom import bench_solid_cohom
from quant_fund.models.solid_tensor2 import (
    bench_solid_tensor2,
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


def bench_clausen_scholze2_family(
    seed: int = _SEED + 3410,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "clausen_scholze2",
            bench_clausen_scholze2(seed),
        )
    )


def bench_solid_cohom_family(seed: int = _SEED + 3411) -> dict[str, float]:
    return _floats(_finite_blob("solid_cohom", bench_solid_cohom(seed)))


def bench_nuclear_space_family(
    seed: int = _SEED + 3412,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "nuclear_space",
            bench_nuclear_space(seed),
        )
    )


def bench_analytic_sheaf_family(
    seed: int = _SEED + 3413,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "analytic_sheaf",
            bench_analytic_sheaf(seed),
        )
    )


def bench_solid_tensor2_family(
    seed: int = _SEED + 3414,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "solid_tensor2",
            bench_solid_tensor2(seed),
        )
    )


def bench_proetale_site2_family(
    seed: int = _SEED + 3415,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "proetale_site2",
            bench_proetale_site2(seed),
        )
    )
