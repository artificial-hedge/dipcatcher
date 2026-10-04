"""Wave-1128 linguistics-6 canon benchmark adapters (SYNTHETIC)."""

from __future__ import annotations

import math
from collections.abc import Iterable

from quant_fund.models.computational_stylistics import bench_computational_stylistics
from quant_fund.models.corpus_phonology import bench_corpus_phonology
from quant_fund.models.language_documentation import bench_language_documentation
from quant_fund.models.lexical_semantics import bench_lexical_semantics
from quant_fund.models.stylistics import bench_stylistics
from quant_fund.models.translation_technology import bench_translation_technology

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


def bench_lexical_semantics_family(seed: int = _SEED + 51600) -> dict[str, float]:
    return _finite_blob(bench_lexical_semantics(seed))


def bench_computational_stylistics_family(seed: int = _SEED + 51601) -> dict[str, float]:
    return _finite_blob(bench_computational_stylistics(seed))


def bench_stylistics_family(seed: int = _SEED + 51602) -> dict[str, float]:
    return _finite_blob(bench_stylistics(seed))


def bench_corpus_phonology_family(seed: int = _SEED + 51603) -> dict[str, float]:
    return _finite_blob(bench_corpus_phonology(seed))


def bench_language_documentation_family(seed: int = _SEED + 51604) -> dict[str, float]:
    return _finite_blob(bench_language_documentation(seed))


def bench_translation_technology_family(seed: int = _SEED + 51605) -> dict[str, float]:
    return _finite_blob(bench_translation_technology(seed))
