"""Wave-1087 documentary-sciences canon benchmark adapters (SYNTHETIC)."""

from __future__ import annotations

import math
from collections.abc import Iterable

from quant_fund.models.diplomatics import bench_diplomatics
from quant_fund.models.epigraphy import bench_epigraphy
from quant_fund.models.genealogy_studies import bench_genealogy_studies
from quant_fund.models.heraldry import bench_heraldry
from quant_fund.models.onomastics import bench_onomastics
from quant_fund.models.sigillography import bench_sigillography

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


def bench_epigraphy_family(seed: int = _SEED + 47500) -> dict[str, float]:
    return _finite_blob(bench_epigraphy(seed))


def bench_diplomatics_family(seed: int = _SEED + 47501) -> dict[str, float]:
    return _finite_blob(bench_diplomatics(seed))


def bench_sigillography_family(seed: int = _SEED + 47502) -> dict[str, float]:
    return _finite_blob(bench_sigillography(seed))


def bench_heraldry_family(seed: int = _SEED + 47503) -> dict[str, float]:
    return _finite_blob(bench_heraldry(seed))


def bench_genealogy_studies_family(seed: int = _SEED + 47504) -> dict[str, float]:
    return _finite_blob(bench_genealogy_studies(seed))


def bench_onomastics_family(seed: int = _SEED + 47505) -> dict[str, float]:
    return _finite_blob(bench_onomastics(seed))
