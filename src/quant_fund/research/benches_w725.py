"""Wave-725 automorphic-points bench adapters (SYNTHETIC-only)."""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.arithmetic_arnold import (
    bench_arithmetic_arnold,
)
from quant_fund.models.bertolini_darmon import (
    bench_bertolini_darmon,
)
from quant_fund.models.darmon_point import bench_darmon_point
from quant_fund.models.howard_main import bench_howard_main
from quant_fund.models.p_group_iwasawa import (
    bench_p_group_iwasawa,
)
from quant_fund.models.shimura_period import (
    bench_shimura_period,
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


def bench_p_group_iwasawa_family(
    seed: int = _SEED + 11400,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "p_group_iwasawa",
            bench_p_group_iwasawa(seed),
        )
    )


def bench_shimura_period_family(
    seed: int = _SEED + 11401,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "shimura_period",
            bench_shimura_period(seed),
        )
    )


def bench_arithmetic_arnold_family(
    seed: int = _SEED + 11402,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "arithmetic_arnold",
            bench_arithmetic_arnold(seed),
        )
    )


def bench_darmon_point_family(
    seed: int = _SEED + 11403,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "darmon_point",
            bench_darmon_point(seed),
        )
    )


def bench_bertolini_darmon_family(
    seed: int = _SEED + 11404,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "bertolini_darmon",
            bench_bertolini_darmon(seed),
        )
    )


def bench_howard_main_family(
    seed: int = _SEED + 11405,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "howard_main",
            bench_howard_main(seed),
        )
    )
