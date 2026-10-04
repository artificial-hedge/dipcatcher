"""Wave-624 formal-geometry bench adapters (SYNTHETIC-only)."""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.adic_formal import bench_adic_formal
from quant_fund.models.algebraization import (
    bench_algebraization,
)
from quant_fund.models.formal_completion import (
    bench_formal_completion,
)
from quant_fund.models.formal_neighborhood import (
    bench_formal_neighborhood,
)
from quant_fund.models.groth_existence import (
    bench_groth_existence,
)
from quant_fund.models.raynaud_formal import (
    bench_raynaud_formal,
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


def bench_raynaud_formal_family(
    seed: int = _SEED + 3650,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "raynaud_formal",
            bench_raynaud_formal(seed),
        )
    )


def bench_formal_completion_family(
    seed: int = _SEED + 3651,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "formal_completion",
            bench_formal_completion(seed),
        )
    )


def bench_adic_formal_family(
    seed: int = _SEED + 3652,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "adic_formal",
            bench_adic_formal(seed),
        )
    )


def bench_formal_neighborhood_family(
    seed: int = _SEED + 3653,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "formal_neighborhood",
            bench_formal_neighborhood(seed),
        )
    )


def bench_groth_existence_family(
    seed: int = _SEED + 3654,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "groth_existence",
            bench_groth_existence(seed),
        )
    )


def bench_algebraization_family(
    seed: int = _SEED + 3655,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "algebraization",
            bench_algebraization(seed),
        )
    )
