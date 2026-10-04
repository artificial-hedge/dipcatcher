"""Wave-588 homotopy-11 bench adapters (SYNTHETIC-only)."""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.ehp_sequence import bench_ehp_sequence
from quant_fund.models.freudenthal_susp import (
    bench_freudenthal_susp,
)
from quant_fund.models.james_period import bench_james_period
from quant_fund.models.moore_space import bench_moore_space
from quant_fund.models.unstable_adams import bench_unstable_adams
from quant_fund.models.whitehead_prod import bench_whitehead_prod

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


def bench_ehp_sequence_family(
    seed: int = _SEED + 3434,
) -> dict[str, float]:
    return _floats(_finite_blob("ehp_sequence", bench_ehp_sequence(seed)))


def bench_james_period_family(
    seed: int = _SEED + 3435,
) -> dict[str, float]:
    return _floats(_finite_blob("james_period", bench_james_period(seed)))


def bench_whitehead_prod_family(
    seed: int = _SEED + 3436,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "whitehead_prod",
            bench_whitehead_prod(seed),
        )
    )


def bench_freudenthal_susp_family(
    seed: int = _SEED + 3437,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "freudenthal_susp",
            bench_freudenthal_susp(seed),
        )
    )


def bench_moore_space_family(
    seed: int = _SEED + 3438,
) -> dict[str, float]:
    return _floats(_finite_blob("moore_space", bench_moore_space(seed)))


def bench_unstable_adams_family(
    seed: int = _SEED + 3439,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "unstable_adams",
            bench_unstable_adams(seed),
        )
    )
