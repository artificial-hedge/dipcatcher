"""Wave-698 motivic-22 bench adapters (SYNTHETIC-only)."""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.motivic_cartier import (
    bench_motivic_cartier,
)
from quant_fund.models.motivic_frobenius import (
    bench_motivic_frobenius,
)
from quant_fund.models.motivic_hodge import (
    bench_motivic_hodge,
)
from quant_fund.models.motivic_lax import bench_motivic_lax
from quant_fund.models.motivic_span import bench_motivic_span
from quant_fund.models.motivic_street import (
    bench_motivic_street,
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


def bench_motivic_frobenius_family(
    seed: int = _SEED + 8700,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "motivic_frobenius",
            bench_motivic_frobenius(seed),
        )
    )


def bench_motivic_cartier_family(
    seed: int = _SEED + 8701,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "motivic_cartier",
            bench_motivic_cartier(seed),
        )
    )


def bench_motivic_hodge_family(
    seed: int = _SEED + 8702,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "motivic_hodge",
            bench_motivic_hodge(seed),
        )
    )


def bench_motivic_span_family(
    seed: int = _SEED + 8703,
) -> dict[str, float]:
    return _floats(_finite_blob("motivic_span", bench_motivic_span(seed)))


def bench_motivic_lax_family(
    seed: int = _SEED + 8704,
) -> dict[str, float]:
    return _floats(_finite_blob("motivic_lax", bench_motivic_lax(seed)))


def bench_motivic_street_family(
    seed: int = _SEED + 8705,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "motivic_street",
            bench_motivic_street(seed),
        )
    )
