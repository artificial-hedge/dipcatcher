"""Wave-1014 astrophysics/cosmology canon benchmark adapters (SYNTHETIC)."""

from __future__ import annotations

import math
from collections.abc import Iterable

from quant_fund.models.cmb_anisotropy import bench_cmb_anisotropy
from quant_fund.models.dark_matter import bench_dark_matter
from quant_fund.models.hubble_law import bench_hubble_law
from quant_fund.models.jeans_instability import bench_jeans_instability
from quant_fund.models.stellar_evolution import bench_stellar_evolution
from quant_fund.models.stellar_structure import bench_stellar_structure

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


def bench_jeans_instability_family(seed: int = _SEED + 40200) -> dict[str, float]:
    return _finite_blob(bench_jeans_instability(seed))


def bench_stellar_structure_family(seed: int = _SEED + 40201) -> dict[str, float]:
    return _finite_blob(bench_stellar_structure(seed))


def bench_stellar_evolution_family(seed: int = _SEED + 40202) -> dict[str, float]:
    return _finite_blob(bench_stellar_evolution(seed))


def bench_hubble_law_family(seed: int = _SEED + 40203) -> dict[str, float]:
    return _finite_blob(bench_hubble_law(seed))


def bench_cmb_anisotropy_family(seed: int = _SEED + 40204) -> dict[str, float]:
    return _finite_blob(bench_cmb_anisotropy(seed))


def bench_dark_matter_family(seed: int = _SEED + 40205) -> dict[str, float]:
    return _finite_blob(bench_dark_matter(seed))
