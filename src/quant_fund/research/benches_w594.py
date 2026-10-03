"""Wave-594 etale-cohomology bench adapters (SYNTHETIC-only)."""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.artin_neighborhood import (
    bench_artin_neighborhood,
)
from quant_fund.models.etale_fund import bench_etale_fund
from quant_fund.models.etale_homotopy import (
    bench_etale_homotopy,
)
from quant_fund.models.galois_cat import bench_galois_cat
from quant_fund.models.pro_etale import bench_pro_etale
from quant_fund.models.shapiro_lemma import (
    bench_shapiro_lemma,
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


def bench_etale_homotopy_family(
    seed: int = _SEED + 3470,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "etale_homotopy",
            bench_etale_homotopy(seed),
        )
    )


def bench_pro_etale_family(
    seed: int = _SEED + 3471,
) -> dict[str, float]:
    return _floats(_finite_blob("pro_etale", bench_pro_etale(seed)))


def bench_etale_fund_family(
    seed: int = _SEED + 3472,
) -> dict[str, float]:
    return _floats(_finite_blob("etale_fund", bench_etale_fund(seed)))


def bench_galois_cat_family(
    seed: int = _SEED + 3473,
) -> dict[str, float]:
    return _floats(_finite_blob("galois_cat", bench_galois_cat(seed)))


def bench_artin_neighborhood_family(
    seed: int = _SEED + 3474,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "artin_neighborhood",
            bench_artin_neighborhood(seed),
        )
    )


def bench_shapiro_lemma_family(
    seed: int = _SEED + 3475,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "shapiro_lemma",
            bench_shapiro_lemma(seed),
        )
    )
