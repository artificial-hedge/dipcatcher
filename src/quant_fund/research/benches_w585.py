"""Wave-585 intersection-theory bench adapters (SYNTHETIC-only)."""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.canonical_bundle import (
    bench_canonical_bundle,
)
from quant_fund.models.intersection_theory import (
    bench_intersection_theory,
)
from quant_fund.models.line_bundle import bench_line_bundle
from quant_fund.models.macpherson_chern import (
    bench_macpherson_chern,
)
from quant_fund.models.picard_group import bench_picard_group
from quant_fund.models.weil_divisor import bench_weil_divisor

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


def bench_intersection_theory_family(
    seed: int = _SEED + 3416,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "intersection_theory",
            bench_intersection_theory(seed),
        )
    )


def bench_macpherson_chern_family(
    seed: int = _SEED + 3417,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "macpherson_chern",
            bench_macpherson_chern(seed),
        )
    )


def bench_weil_divisor_family(
    seed: int = _SEED + 3418,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "weil_divisor",
            bench_weil_divisor(seed),
        )
    )


def bench_picard_group_family(
    seed: int = _SEED + 3419,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "picard_group",
            bench_picard_group(seed),
        )
    )


def bench_line_bundle_family(seed: int = _SEED + 3420) -> dict[str, float]:
    return _floats(_finite_blob("line_bundle", bench_line_bundle(seed)))


def bench_canonical_bundle_family(
    seed: int = _SEED + 3421,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "canonical_bundle",
            bench_canonical_bundle(seed),
        )
    )
