"""Wave-1079 classics canon benchmark adapters (SYNTHETIC)."""

from __future__ import annotations

import math
from collections.abc import Iterable

from quant_fund.models.ancient_greek import bench_ancient_greek
from quant_fund.models.classical_archaeology import bench_classical_archaeology
from quant_fund.models.classical_studies import bench_classical_studies
from quant_fund.models.latin_language import bench_latin_language
from quant_fund.models.papyrology import bench_papyrology
from quant_fund.models.philology import bench_philology

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


def bench_classical_studies_family(seed: int = _SEED + 46700) -> dict[str, float]:
    return _finite_blob(bench_classical_studies(seed))


def bench_latin_language_family(seed: int = _SEED + 46701) -> dict[str, float]:
    return _finite_blob(bench_latin_language(seed))


def bench_ancient_greek_family(seed: int = _SEED + 46702) -> dict[str, float]:
    return _finite_blob(bench_ancient_greek(seed))


def bench_classical_archaeology_family(seed: int = _SEED + 46703) -> dict[str, float]:
    return _finite_blob(bench_classical_archaeology(seed))


def bench_philology_family(seed: int = _SEED + 46704) -> dict[str, float]:
    return _finite_blob(bench_philology(seed))


def bench_papyrology_family(seed: int = _SEED + 46705) -> dict[str, float]:
    return _finite_blob(bench_papyrology(seed))
