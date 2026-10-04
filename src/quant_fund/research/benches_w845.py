"""Wave-845 integral-transforms bench adapters (SYNTHETIC-only)."""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.abel_transform import (
    bench_abel_transform,
)
from quant_fund.models.hankel_transform import (
    bench_hankel_transform,
)
from quant_fund.models.hilbert_transform import (
    bench_hilbert_transform,
)
from quant_fund.models.laplace_transform import (
    bench_laplace_transform,
)
from quant_fund.models.mellin_transform import (
    bench_mellin_transform,
)
from quant_fund.models.z_transform import (
    bench_z_transform,
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


def bench_laplace_transform_family(
    seed: int = _SEED + 23300,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "laplace_transform",
            bench_laplace_transform(seed),
        )
    )


def bench_mellin_transform_family(
    seed: int = _SEED + 23301,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "mellin_transform",
            bench_mellin_transform(seed),
        )
    )


def bench_hankel_transform_family(
    seed: int = _SEED + 23302,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "hankel_transform",
            bench_hankel_transform(seed),
        )
    )


def bench_z_transform_family(
    seed: int = _SEED + 23303,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "z_transform",
            bench_z_transform(seed),
        )
    )


def bench_hilbert_transform_family(
    seed: int = _SEED + 23304,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "hilbert_transform",
            bench_hilbert_transform(seed),
        )
    )


def bench_abel_transform_family(
    seed: int = _SEED + 23305,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "abel_transform",
            bench_abel_transform(seed),
        )
    )
