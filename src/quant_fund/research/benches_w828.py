"""Wave-828 projection/section bench adapters (SYNTHETIC-only)."""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.cross_section import (
    bench_cross_section,
)
from quant_fund.models.dellacherie_section import (
    bench_dellacherie_section,
)
from quant_fund.models.maharam_lift import (
    bench_maharam_lift,
)
from quant_fund.models.projection_theorem import (
    bench_projection_theorem,
)
from quant_fund.models.uniform_section import (
    bench_uniform_section,
)
from quant_fund.models.von_neumann_sel import (
    bench_von_neumann_sel,
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


def bench_projection_theorem_family(
    seed: int = _SEED + 21600,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "projection_theorem",
            bench_projection_theorem(seed),
        )
    )


def bench_uniform_section_family(
    seed: int = _SEED + 21601,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "uniform_section",
            bench_uniform_section(seed),
        )
    )


def bench_dellacherie_section_family(
    seed: int = _SEED + 21602,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "dellacherie_section",
            bench_dellacherie_section(seed),
        )
    )


def bench_cross_section_family(
    seed: int = _SEED + 21603,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "cross_section",
            bench_cross_section(seed),
        )
    )


def bench_maharam_lift_family(
    seed: int = _SEED + 21604,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "maharam_lift",
            bench_maharam_lift(seed),
        )
    )


def bench_von_neumann_sel_family(
    seed: int = _SEED + 21605,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "von_neumann_sel",
            bench_von_neumann_sel(seed),
        )
    )
