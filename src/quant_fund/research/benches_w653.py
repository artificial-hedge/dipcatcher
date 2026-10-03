"""Wave-653 arithmetic-geometry-2 bench adapters (SYNTHETIC-only)."""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.ainf_cohom import bench_ainf_cohom
from quant_fund.models.fargues_scholze3 import bench_fargues_scholze3
from quant_fund.models.galois_padic import bench_galois_padic
from quant_fund.models.hodge_tate_padic import bench_hodge_tate_padic
from quant_fund.models.integral_padic2 import bench_integral_padic2
from quant_fund.models.period_ring import bench_period_ring

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


def bench_fargues_scholze3_family(
    seed: int = _SEED + 4200,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "fargues_scholze3",
            bench_fargues_scholze3(seed),
        )
    )


def bench_integral_padic2_family(
    seed: int = _SEED + 4201,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "integral_padic2",
            bench_integral_padic2(seed),
        )
    )


def bench_ainf_cohom_family(
    seed: int = _SEED + 4202,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "ainf_cohom",
            bench_ainf_cohom(seed),
        )
    )


def bench_period_ring_family(
    seed: int = _SEED + 4203,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "period_ring",
            bench_period_ring(seed),
        )
    )


def bench_galois_padic_family(
    seed: int = _SEED + 4204,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "galois_padic",
            bench_galois_padic(seed),
        )
    )


def bench_hodge_tate_padic_family(
    seed: int = _SEED + 4205,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "hodge_tate_padic",
            bench_hodge_tate_padic(seed),
        )
    )
