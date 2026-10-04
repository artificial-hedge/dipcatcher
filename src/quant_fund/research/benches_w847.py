"""Wave-847 perturbation-theory bench adapters (SYNTHETIC-only)."""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.boundary_layer import (
    bench_boundary_layer,
)
from quant_fund.models.lindstedt_poincare import (
    bench_lindstedt_poincare,
)
from quant_fund.models.matched_asymptotic import (
    bench_matched_asymptotic,
)
from quant_fund.models.multiple_scales import (
    bench_multiple_scales,
)
from quant_fund.models.regular_perturbation import (
    bench_regular_perturbation,
)
from quant_fund.models.singular_perturbation import (
    bench_singular_perturbation,
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


def bench_regular_perturbation_family(
    seed: int = _SEED + 23500,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "regular_perturbation",
            bench_regular_perturbation(seed),
        )
    )


def bench_singular_perturbation_family(
    seed: int = _SEED + 23501,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "singular_perturbation",
            bench_singular_perturbation(seed),
        )
    )


def bench_matched_asymptotic_family(
    seed: int = _SEED + 23502,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "matched_asymptotic",
            bench_matched_asymptotic(seed),
        )
    )


def bench_multiple_scales_family(
    seed: int = _SEED + 23503,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "multiple_scales",
            bench_multiple_scales(seed),
        )
    )


def bench_lindstedt_poincare_family(
    seed: int = _SEED + 23504,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "lindstedt_poincare",
            bench_lindstedt_poincare(seed),
        )
    )


def bench_boundary_layer_family(
    seed: int = _SEED + 23505,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "boundary_layer",
            bench_boundary_layer(seed),
        )
    )
