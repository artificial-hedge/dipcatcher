"""Wave-1082 jewish studies canon benchmark adapters (SYNTHETIC)."""

from __future__ import annotations

import math
from collections.abc import Iterable

from quant_fund.models.hebrew_language import bench_hebrew_language
from quant_fund.models.jewish_philosophy import bench_jewish_philosophy
from quant_fund.models.jewish_studies import bench_jewish_studies
from quant_fund.models.kabbalah import bench_kabbalah
from quant_fund.models.rabbinics import bench_rabbinics
from quant_fund.models.talmudic_studies import bench_talmudic_studies

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


def bench_jewish_studies_family(seed: int = _SEED + 47000) -> dict[str, float]:
    return _finite_blob(bench_jewish_studies(seed))


def bench_talmudic_studies_family(seed: int = _SEED + 47001) -> dict[str, float]:
    return _finite_blob(bench_talmudic_studies(seed))


def bench_hebrew_language_family(seed: int = _SEED + 47002) -> dict[str, float]:
    return _finite_blob(bench_hebrew_language(seed))


def bench_rabbinics_family(seed: int = _SEED + 47003) -> dict[str, float]:
    return _finite_blob(bench_rabbinics(seed))


def bench_kabbalah_family(seed: int = _SEED + 47004) -> dict[str, float]:
    return _finite_blob(bench_kabbalah(seed))


def bench_jewish_philosophy_family(seed: int = _SEED + 47005) -> dict[str, float]:
    return _finite_blob(bench_jewish_philosophy(seed))
