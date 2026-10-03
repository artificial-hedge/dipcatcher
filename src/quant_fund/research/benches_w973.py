"""Wave-973 Banach-space-geometry canon benchmark adapters (SYNTHETIC)."""

from __future__ import annotations

import math
from collections.abc import Iterable

from quant_fund.models.banach_mazur import bench_banach_mazur
from quant_fund.models.djt_space import bench_djt_space
from quant_fund.models.gl_property import bench_gl_property
from quant_fund.models.kalton_loc import bench_kalton_loc
from quant_fund.models.schauder_basis import bench_schauder_basis
from quant_fund.models.type_cotype import bench_type_cotype

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


def bench_banach_mazur_family(seed: int = _SEED + 36100) -> dict[str, float]:
    return _finite_blob(bench_banach_mazur(seed))


def bench_type_cotype_family(seed: int = _SEED + 36101) -> dict[str, float]:
    return _finite_blob(bench_type_cotype(seed))


def bench_gl_property_family(seed: int = _SEED + 36102) -> dict[str, float]:
    return _finite_blob(bench_gl_property(seed))


def bench_djt_space_family(seed: int = _SEED + 36103) -> dict[str, float]:
    return _finite_blob(bench_djt_space(seed))


def bench_schauder_basis_family(seed: int = _SEED + 36104) -> dict[str, float]:
    return _finite_blob(bench_schauder_basis(seed))


def bench_kalton_loc_family(seed: int = _SEED + 36105) -> dict[str, float]:
    return _finite_blob(bench_kalton_loc(seed))
