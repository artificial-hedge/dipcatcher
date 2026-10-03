"""Wave-627 condensed-4 bench adapters (SYNTHETIC-only)."""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.condensed_coh import (
    bench_condensed_coh,
)
from quant_fund.models.condensed_ring import (
    bench_condensed_ring,
)
from quant_fund.models.discrete_liquid import (
    bench_discrete_liquid,
)
from quant_fund.models.liquid_ring import (
    bench_liquid_ring,
)
from quant_fund.models.scholze_trace import (
    bench_scholze_trace,
)
from quant_fund.models.smith_project import (
    bench_smith_project,
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


def bench_discrete_liquid_family(
    seed: int = _SEED + 3668,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "discrete_liquid",
            bench_discrete_liquid(seed),
        )
    )


def bench_smith_project_family(
    seed: int = _SEED + 3669,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "smith_project",
            bench_smith_project(seed),
        )
    )


def bench_condensed_ring_family(
    seed: int = _SEED + 3670,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "condensed_ring",
            bench_condensed_ring(seed),
        )
    )


def bench_liquid_ring_family(
    seed: int = _SEED + 3671,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "liquid_ring",
            bench_liquid_ring(seed),
        )
    )


def bench_scholze_trace_family(
    seed: int = _SEED + 3672,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "scholze_trace",
            bench_scholze_trace(seed),
        )
    )


def bench_condensed_coh_family(
    seed: int = _SEED + 3673,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "condensed_coh",
            bench_condensed_coh(seed),
        )
    )
