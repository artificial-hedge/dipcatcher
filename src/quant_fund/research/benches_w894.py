"""Wave-894 interpolation bench adapters (SYNTHETIC-only)."""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.barycentric_wts import (
    bench_barycentric_wts,
)
from quant_fund.models.divid_diff_table import (
    bench_divid_diff_table,
)
from quant_fund.models.floater_hormann import (
    bench_floater_hormann,
)
from quant_fund.models.hermite_interp import (
    bench_hermite_interp,
)
from quant_fund.models.lagrange_interp import (
    bench_lagrange_interp,
)
from quant_fund.models.neville_interp import (
    bench_neville_interp,
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


def bench_lagrange_interp_family(
    seed: int = _SEED + 28200,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "lagrange_interp",
            bench_lagrange_interp(seed),
        )
    )


def bench_neville_interp_family(
    seed: int = _SEED + 28201,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "neville_interp",
            bench_neville_interp(seed),
        )
    )


def bench_hermite_interp_family(
    seed: int = _SEED + 28202,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "hermite_interp",
            bench_hermite_interp(seed),
        )
    )


def bench_divid_diff_table_family(
    seed: int = _SEED + 28203,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "divid_diff_table",
            bench_divid_diff_table(seed),
        )
    )


def bench_barycentric_wts_family(
    seed: int = _SEED + 28204,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "barycentric_wts",
            bench_barycentric_wts(seed),
        )
    )


def bench_floater_hormann_family(
    seed: int = _SEED + 28205,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "floater_hormann",
            bench_floater_hormann(seed),
        )
    )
