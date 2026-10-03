"""Wave-823 law-of-process bench adapters (SYNTHETIC-only)."""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.cylindrical_law import (
    bench_cylindrical_law,
)
from quant_fund.models.finite_dim import (
    bench_finite_dim,
)
from quant_fund.models.law_convergence import (
    bench_law_convergence,
)
from quant_fund.models.polish_law import (
    bench_polish_law,
)
from quant_fund.models.support_law import (
    bench_support_law,
)
from quant_fund.models.tight_law import (
    bench_tight_law,
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


def bench_support_law_family(
    seed: int = _SEED + 21100,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "support_law",
            bench_support_law(seed),
        )
    )


def bench_polish_law_family(
    seed: int = _SEED + 21101,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "polish_law",
            bench_polish_law(seed),
        )
    )


def bench_tight_law_family(
    seed: int = _SEED + 21102,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "tight_law",
            bench_tight_law(seed),
        )
    )


def bench_law_convergence_family(
    seed: int = _SEED + 21103,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "law_convergence",
            bench_law_convergence(seed),
        )
    )


def bench_finite_dim_family(
    seed: int = _SEED + 21104,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "finite_dim",
            bench_finite_dim(seed),
        )
    )


def bench_cylindrical_law_family(
    seed: int = _SEED + 21105,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "cylindrical_law",
            bench_cylindrical_law(seed),
        )
    )
