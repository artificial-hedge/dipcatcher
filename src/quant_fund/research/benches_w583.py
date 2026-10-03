"""Wave-583 motivic-6 bench adapters (SYNTHETIC-only)."""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.fulton_mclarty import (
    bench_fulton_mclarty,
)
from quant_fund.models.motivic_base_change import (
    bench_motivic_base_change,
)
from quant_fund.models.motivic_homotopy2 import (
    bench_motivic_homotopy2,
)
from quant_fund.models.motivic_proper import (
    bench_motivic_proper,
)
from quant_fund.models.motivic_smooth import (
    bench_motivic_smooth,
)
from quant_fund.models.six_op_motivic import (
    bench_six_op_motivic,
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


def bench_motivic_base_change_family(
    seed: int = _SEED + 3404,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "motivic_base_change",
            bench_motivic_base_change(seed),
        )
    )


def bench_six_op_motivic_family(
    seed: int = _SEED + 3405,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "six_op_motivic",
            bench_six_op_motivic(seed),
        )
    )


def bench_motivic_smooth_family(
    seed: int = _SEED + 3406,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "motivic_smooth",
            bench_motivic_smooth(seed),
        )
    )


def bench_motivic_proper_family(
    seed: int = _SEED + 3407,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "motivic_proper",
            bench_motivic_proper(seed),
        )
    )


def bench_fulton_mclarty_family(
    seed: int = _SEED + 3408,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "fulton_mclarty",
            bench_fulton_mclarty(seed),
        )
    )


def bench_motivic_homotopy2_family(
    seed: int = _SEED + 3409,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "motivic_homotopy2",
            bench_motivic_homotopy2(seed),
        )
    )
