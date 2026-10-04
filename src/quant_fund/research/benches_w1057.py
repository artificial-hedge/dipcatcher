"""Wave-1057 linguistics canon benchmark adapters (SYNTHETIC)."""

from __future__ import annotations

import math
from collections.abc import Iterable

from quant_fund.models.morphology import bench_morphology
from quant_fund.models.phonetics import bench_phonetics
from quant_fund.models.phonology import bench_phonology
from quant_fund.models.pragmatics import bench_pragmatics
from quant_fund.models.semantics import bench_semantics
from quant_fund.models.syntax_theory import bench_syntax_theory

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


def bench_phonetics_family(seed: int = _SEED + 44500) -> dict[str, float]:
    return _finite_blob(bench_phonetics(seed))


def bench_phonology_family(seed: int = _SEED + 44501) -> dict[str, float]:
    return _finite_blob(bench_phonology(seed))


def bench_morphology_family(seed: int = _SEED + 44502) -> dict[str, float]:
    return _finite_blob(bench_morphology(seed))


def bench_syntax_theory_family(seed: int = _SEED + 44503) -> dict[str, float]:
    return _finite_blob(bench_syntax_theory(seed))


def bench_semantics_family(seed: int = _SEED + 44504) -> dict[str, float]:
    return _finite_blob(bench_semantics(seed))


def bench_pragmatics_family(seed: int = _SEED + 44505) -> dict[str, float]:
    return _finite_blob(bench_pragmatics(seed))
