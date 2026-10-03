"""Wave-623 higher-operads bench adapters (SYNTHETIC-only)."""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.a_infinity2 import bench_a_infinity2
from quant_fund.models.cyclic_operad import (
    bench_cyclic_operad,
)
from quant_fund.models.dendroidal2 import bench_dendroidal2
from quant_fund.models.e_infinity3 import bench_e_infinity3
from quant_fund.models.infty_operad2 import (
    bench_infty_operad2,
)
from quant_fund.models.operadic_nerve import (
    bench_operadic_nerve,
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


def bench_dendroidal2_family(
    seed: int = _SEED + 3644,
) -> dict[str, float]:
    return _floats(_finite_blob("dendroidal2", bench_dendroidal2(seed)))


def bench_operadic_nerve_family(
    seed: int = _SEED + 3645,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "operadic_nerve",
            bench_operadic_nerve(seed),
        )
    )


def bench_infty_operad2_family(
    seed: int = _SEED + 3646,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "infty_operad2",
            bench_infty_operad2(seed),
        )
    )


def bench_a_infinity2_family(
    seed: int = _SEED + 3647,
) -> dict[str, float]:
    return _floats(_finite_blob("a_infinity2", bench_a_infinity2(seed)))


def bench_e_infinity3_family(
    seed: int = _SEED + 3648,
) -> dict[str, float]:
    return _floats(_finite_blob("e_infinity3", bench_e_infinity3(seed)))


def bench_cyclic_operad_family(
    seed: int = _SEED + 3649,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "cyclic_operad",
            bench_cyclic_operad(seed),
        )
    )
