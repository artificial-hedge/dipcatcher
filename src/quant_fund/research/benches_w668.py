"""Wave-668 motivic-15 bench adapters (SYNTHETIC-only)."""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.beilinson_regulator2 import (
    bench_beilinson_regulator2,
)
from quant_fund.models.hodge_motive2 import bench_hodge_motive2
from quant_fund.models.motivic_galois2 import (
    bench_motivic_galois2,
)
from quant_fund.models.norimotive3 import bench_norimotive3
from quant_fund.models.period_realization2 import (
    bench_period_realization2,
)
from quant_fund.models.tannakian_motive2 import (
    bench_tannakian_motive2,
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


def bench_norimotive3_family(
    seed: int = _SEED + 5700,
) -> dict[str, float]:
    return _floats(_finite_blob("norimotive3", bench_norimotive3(seed)))


def bench_motivic_galois2_family(
    seed: int = _SEED + 5701,
) -> dict[str, float]:
    return _floats(_finite_blob("motivic_galois2", bench_motivic_galois2(seed)))


def bench_tannakian_motive2_family(
    seed: int = _SEED + 5702,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "tannakian_motive2",
            bench_tannakian_motive2(seed),
        )
    )


def bench_period_realization2_family(
    seed: int = _SEED + 5703,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "period_realization2",
            bench_period_realization2(seed),
        )
    )


def bench_beilinson_regulator2_family(
    seed: int = _SEED + 5704,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "beilinson_regulator2",
            bench_beilinson_regulator2(seed),
        )
    )


def bench_hodge_motive2_family(
    seed: int = _SEED + 5705,
) -> dict[str, float]:
    return _floats(_finite_blob("hodge_motive2", bench_hodge_motive2(seed)))
