"""Wave-703 homotopy-30 bench adapters (SYNTHETIC-only)."""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.homotopy_fiber3 import (
    bench_homotopy_fiber3,
)
from quant_fund.models.homotopy_spectrum2 import (
    bench_homotopy_spectrum2,
)
from quant_fund.models.homotopy_suspension2 import (
    bench_homotopy_suspension2,
)
from quant_fund.models.homotopy_vn import bench_homotopy_vn
from quant_fund.models.stable_derivator import (
    bench_stable_derivator,
)
from quant_fund.models.stable_excisive import (
    bench_stable_excisive,
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


def bench_homotopy_suspension2_family(
    seed: int = _SEED + 9200,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "homotopy_suspension2",
            bench_homotopy_suspension2(seed),
        )
    )


def bench_homotopy_fiber3_family(
    seed: int = _SEED + 9201,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "homotopy_fiber3",
            bench_homotopy_fiber3(seed),
        )
    )


def bench_stable_derivator_family(
    seed: int = _SEED + 9202,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "stable_derivator",
            bench_stable_derivator(seed),
        )
    )


def bench_homotopy_spectrum2_family(
    seed: int = _SEED + 9203,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "homotopy_spectrum2",
            bench_homotopy_spectrum2(seed),
        )
    )


def bench_stable_excisive_family(
    seed: int = _SEED + 9204,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "stable_excisive",
            bench_stable_excisive(seed),
        )
    )


def bench_homotopy_vn_family(
    seed: int = _SEED + 9205,
) -> dict[str, float]:
    return _floats(_finite_blob("homotopy_vn", bench_homotopy_vn(seed)))
