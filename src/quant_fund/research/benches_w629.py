"""Wave-629 deformations-3 bench adapters (SYNTHETIC-only)."""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.artinian_alg import (
    bench_artinian_alg,
)
from quant_fund.models.deform_functor2 import (
    bench_deform_functor2,
)
from quant_fund.models.hull_deform import (
    bench_hull_deform,
)
from quant_fund.models.rim_deform import (
    bench_rim_deform,
)
from quant_fund.models.small_ext import bench_small_ext
from quant_fund.models.tangent_def import (
    bench_tangent_def,
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


def bench_deform_functor2_family(
    seed: int = _SEED + 3680,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "deform_functor2",
            bench_deform_functor2(seed),
        )
    )


def bench_tangent_def_family(
    seed: int = _SEED + 3681,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "tangent_def",
            bench_tangent_def(seed),
        )
    )


def bench_rim_deform_family(
    seed: int = _SEED + 3682,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "rim_deform",
            bench_rim_deform(seed),
        )
    )


def bench_small_ext_family(
    seed: int = _SEED + 3683,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "small_ext",
            bench_small_ext(seed),
        )
    )


def bench_hull_deform_family(
    seed: int = _SEED + 3684,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "hull_deform",
            bench_hull_deform(seed),
        )
    )


def bench_artinian_alg_family(
    seed: int = _SEED + 3685,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "artinian_alg",
            bench_artinian_alg(seed),
        )
    )
