"""Wave-835 stochastic-geometry bench adapters (SYNTHETIC-only)."""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.boolean_model import (
    bench_boolean_model,
)
from quant_fund.models.germ_grain import (
    bench_germ_grain,
)
from quant_fund.models.intrinsic_volumes import (
    bench_intrinsic_volumes,
)
from quant_fund.models.miles_matheron import (
    bench_miles_matheron,
)
from quant_fund.models.poisson_voronoi import (
    bench_poisson_voronoi,
)
from quant_fund.models.steiner_formula import (
    bench_steiner_formula,
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


def bench_poisson_voronoi_family(
    seed: int = _SEED + 22300,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "poisson_voronoi",
            bench_poisson_voronoi(seed),
        )
    )


def bench_boolean_model_family(
    seed: int = _SEED + 22301,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "boolean_model",
            bench_boolean_model(seed),
        )
    )


def bench_germ_grain_family(
    seed: int = _SEED + 22302,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "germ_grain",
            bench_germ_grain(seed),
        )
    )


def bench_steiner_formula_family(
    seed: int = _SEED + 22303,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "steiner_formula",
            bench_steiner_formula(seed),
        )
    )


def bench_miles_matheron_family(
    seed: int = _SEED + 22304,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "miles_matheron",
            bench_miles_matheron(seed),
        )
    )


def bench_intrinsic_volumes_family(
    seed: int = _SEED + 22305,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "intrinsic_volumes",
            bench_intrinsic_volumes(seed),
        )
    )
