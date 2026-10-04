"""Wave-669 motivic-16 bench adapters (SYNTHETIC-only)."""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.f_motive2 import bench_f_motive2
from quant_fund.models.milnor_operations2 import (
    bench_milnor_operations2,
)
from quant_fund.models.motivic_bordism import (
    bench_motivic_bordism,
)
from quant_fund.models.motivic_eilenberg2 import (
    bench_motivic_eilenberg2,
)
from quant_fund.models.motivic_ss2 import bench_motivic_ss2
from quant_fund.models.slice_filtration2 import (
    bench_slice_filtration2,
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


def bench_slice_filtration2_family(
    seed: int = _SEED + 5800,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "slice_filtration2",
            bench_slice_filtration2(seed),
        )
    )


def bench_milnor_operations2_family(
    seed: int = _SEED + 5801,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "milnor_operations2",
            bench_milnor_operations2(seed),
        )
    )


def bench_motivic_bordism_family(
    seed: int = _SEED + 5802,
) -> dict[str, float]:
    return _floats(_finite_blob("motivic_bordism", bench_motivic_bordism(seed)))


def bench_motivic_eilenberg2_family(
    seed: int = _SEED + 5803,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "motivic_eilenberg2",
            bench_motivic_eilenberg2(seed),
        )
    )


def bench_f_motive2_family(
    seed: int = _SEED + 5804,
) -> dict[str, float]:
    return _floats(_finite_blob("f_motive2", bench_f_motive2(seed)))


def bench_motivic_ss2_family(
    seed: int = _SEED + 5805,
) -> dict[str, float]:
    return _floats(_finite_blob("motivic_ss2", bench_motivic_ss2(seed)))
