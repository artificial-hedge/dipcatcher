"""Wave-900 spatial-index bench adapters (SYNTHETIC-only)."""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.ball_tree import (
    bench_ball_tree,
)
from quant_fund.models.cover_tree import (
    bench_cover_tree,
)
from quant_fund.models.kd_tree import (
    bench_kd_tree,
)
from quant_fund.models.quad_tree import (
    bench_quad_tree,
)
from quant_fund.models.r_tree import (
    bench_r_tree,
)
from quant_fund.models.vp_tree import (
    bench_vp_tree,
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


def bench_kd_tree_family(
    seed: int = _SEED + 28800,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "kd_tree",
            bench_kd_tree(seed),
        )
    )


def bench_ball_tree_family(
    seed: int = _SEED + 28801,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "ball_tree",
            bench_ball_tree(seed),
        )
    )


def bench_cover_tree_family(
    seed: int = _SEED + 28802,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "cover_tree",
            bench_cover_tree(seed),
        )
    )


def bench_r_tree_family(
    seed: int = _SEED + 28803,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "r_tree",
            bench_r_tree(seed),
        )
    )


def bench_quad_tree_family(
    seed: int = _SEED + 28804,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "quad_tree",
            bench_quad_tree(seed),
        )
    )


def bench_vp_tree_family(
    seed: int = _SEED + 28805,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "vp_tree",
            bench_vp_tree(seed),
        )
    )
