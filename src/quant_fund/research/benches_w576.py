"""Wave-576 free-probability bench adapters (SYNTHETIC-only)."""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.free_convolution import (
    bench_free_convolution,
)
from quant_fund.models.free_prob import bench_free_prob
from quant_fund.models.operator_valued import (
    bench_operator_valued,
)
from quant_fund.models.r_transform import bench_r_transform
from quant_fund.models.s_transform import bench_s_transform
from quant_fund.models.voiculescu_thm import (
    bench_voiculescu_thm,
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


def bench_free_prob_family(seed: int = _SEED + 3362) -> dict[str, float]:
    return _floats(_finite_blob("free_prob", bench_free_prob(seed)))


def bench_r_transform_family(seed: int = _SEED + 3363) -> dict[str, float]:
    return _floats(_finite_blob("r_transform", bench_r_transform(seed)))


def bench_s_transform_family(seed: int = _SEED + 3364) -> dict[str, float]:
    return _floats(_finite_blob("s_transform", bench_s_transform(seed)))


def bench_free_convolution_family(
    seed: int = _SEED + 3365,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "free_convolution",
            bench_free_convolution(seed),
        )
    )


def bench_voiculescu_thm_family(
    seed: int = _SEED + 3366,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "voiculescu_thm",
            bench_voiculescu_thm(seed),
        )
    )


def bench_operator_valued_family(
    seed: int = _SEED + 3367,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "operator_valued",
            bench_operator_valued(seed),
        )
    )
