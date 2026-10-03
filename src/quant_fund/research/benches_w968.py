"""Wave-968 KK-theory canon benchmark adapters (SYNTHETIC)."""

from __future__ import annotations

import math
from collections.abc import Iterable

from quant_fund.models.baaj_julg import bench_baaj_julg
from quant_fund.models.cuntz_picture import bench_cuntz_picture
from quant_fund.models.ext_functor import bench_ext_functor
from quant_fund.models.kasparov_prod import bench_kasparov_prod
from quant_fund.models.kk_duality import bench_kk_duality
from quant_fund.models.kk_theory import bench_kk_theory

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 20261231


def _finite_blob(blob: dict[str, float]) -> dict[str, float]:
    out: dict[str, float] = {}
    for key, val in blob.items():
        if key.lower() in _FORBIDDEN:
            raise ValueError(f"forbidden metric key: {key}")
        if not math.isfinite(val):
            raise ValueError(f"non-finite metric: {key}")
        out[key] = float(val)
    return out


def _floats(xs: Iterable[float]) -> list[float]:
    return [float(x) for x in xs]


def bench_kk_theory_family(seed: int = _SEED + 35600) -> dict[str, float]:
    return _finite_blob(bench_kk_theory(seed))


def bench_kasparov_prod_family(seed: int = _SEED + 35601) -> dict[str, float]:
    return _finite_blob(bench_kasparov_prod(seed))


def bench_ext_functor_family(seed: int = _SEED + 35602) -> dict[str, float]:
    return _finite_blob(bench_ext_functor(seed))


def bench_baaj_julg_family(seed: int = _SEED + 35603) -> dict[str, float]:
    return _finite_blob(bench_baaj_julg(seed))


def bench_cuntz_picture_family(seed: int = _SEED + 35604) -> dict[str, float]:
    return _finite_blob(bench_cuntz_picture(seed))


def bench_kk_duality_family(seed: int = _SEED + 35605) -> dict[str, float]:
    return _finite_blob(bench_kk_duality(seed))
