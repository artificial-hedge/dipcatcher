"""Wave-625 homotopy-15 bench adapters (SYNTHETIC-only)."""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.complexity_spectrum import (
    bench_complexity_spectrum,
)
from quant_fund.models.simplicial_htpy import (
    bench_simplicial_htpy,
)
from quant_fund.models.small_spec import bench_small_spec
from quant_fund.models.spectrum_type import (
    bench_spectrum_type,
)
from quant_fund.models.stable_cohomology2 import (
    bench_stable_cohomology2,
)
from quant_fund.models.woodward_op import (
    bench_woodward_op,
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


def bench_stable_cohomology2_family(
    seed: int = _SEED + 3656,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "stable_cohomology2",
            bench_stable_cohomology2(seed),
        )
    )


def bench_woodward_op_family(
    seed: int = _SEED + 3657,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "woodward_op",
            bench_woodward_op(seed),
        )
    )


def bench_spectrum_type_family(
    seed: int = _SEED + 3658,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "spectrum_type",
            bench_spectrum_type(seed),
        )
    )


def bench_complexity_spectrum_family(
    seed: int = _SEED + 3659,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "complexity_spectrum",
            bench_complexity_spectrum(seed),
        )
    )


def bench_small_spec_family(
    seed: int = _SEED + 3660,
) -> dict[str, float]:
    return _floats(_finite_blob("small_spec", bench_small_spec(seed)))


def bench_simplicial_htpy_family(
    seed: int = _SEED + 3661,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "simplicial_htpy",
            bench_simplicial_htpy(seed),
        )
    )
