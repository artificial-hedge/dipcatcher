"""Wave-626 stacks-3 bench adapters (SYNTHETIC-only)."""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.algebraic_stack2 import (
    bench_algebraic_stack2,
)
from quant_fund.models.artin_stack import bench_artin_stack
from quant_fund.models.gerbe_cohomology import (
    bench_gerbe_cohomology,
)
from quant_fund.models.orbifold_stack import (
    bench_orbifold_stack,
)
from quant_fund.models.quotient_stack2 import (
    bench_quotient_stack2,
)
from quant_fund.models.stacky_point import (
    bench_stacky_point,
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


def bench_algebraic_stack2_family(
    seed: int = _SEED + 3662,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "algebraic_stack2",
            bench_algebraic_stack2(seed),
        )
    )


def bench_artin_stack_family(
    seed: int = _SEED + 3663,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "artin_stack",
            bench_artin_stack(seed),
        )
    )


def bench_quotient_stack2_family(
    seed: int = _SEED + 3664,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "quotient_stack2",
            bench_quotient_stack2(seed),
        )
    )


def bench_stacky_point_family(
    seed: int = _SEED + 3665,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "stacky_point",
            bench_stacky_point(seed),
        )
    )


def bench_orbifold_stack_family(
    seed: int = _SEED + 3666,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "orbifold_stack",
            bench_orbifold_stack(seed),
        )
    )


def bench_gerbe_cohomology_family(
    seed: int = _SEED + 3667,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "gerbe_cohomology",
            bench_gerbe_cohomology(seed),
        )
    )
