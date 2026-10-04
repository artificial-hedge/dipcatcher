"""Wave-843 spline-theory bench adapters (SYNTHETIC-only)."""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.b_spline import (
    bench_b_spline,
)
from quant_fund.models.blossoming import (
    bench_blossoming,
)
from quant_fund.models.box_spline import (
    bench_box_spline,
)
from quant_fund.models.cardinal_spline import (
    bench_cardinal_spline,
)
from quant_fund.models.de_boor import (
    bench_de_boor,
)
from quant_fund.models.knot_insertion import (
    bench_knot_insertion,
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


def bench_b_spline_family(
    seed: int = _SEED + 23100,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "b_spline",
            bench_b_spline(seed),
        )
    )


def bench_de_boor_family(
    seed: int = _SEED + 23101,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "de_boor",
            bench_de_boor(seed),
        )
    )


def bench_cardinal_spline_family(
    seed: int = _SEED + 23102,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "cardinal_spline",
            bench_cardinal_spline(seed),
        )
    )


def bench_knot_insertion_family(
    seed: int = _SEED + 23103,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "knot_insertion",
            bench_knot_insertion(seed),
        )
    )


def bench_blossoming_family(
    seed: int = _SEED + 23104,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "blossoming",
            bench_blossoming(seed),
        )
    )


def bench_box_spline_family(
    seed: int = _SEED + 23105,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "box_spline",
            bench_box_spline(seed),
        )
    )
