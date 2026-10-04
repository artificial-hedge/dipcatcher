"""Wave-1118 linguistics-4 canon benchmark adapters (SYNTHETIC)."""

from __future__ import annotations

import math
from collections.abc import Iterable

from quant_fund.models.field_linguistics import bench_field_linguistics
from quant_fund.models.language_acquisition import bench_language_acquisition
from quant_fund.models.linguistic_typology import bench_linguistic_typology
from quant_fund.models.sign_linguistics import bench_sign_linguistics
from quant_fund.models.theoretical_linguistics import bench_theoretical_linguistics
from quant_fund.models.translation_theory import bench_translation_theory

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


def bench_theoretical_linguistics_family(seed: int = _SEED + 50600) -> dict[str, float]:
    return _finite_blob(bench_theoretical_linguistics(seed))


def bench_field_linguistics_family(seed: int = _SEED + 50601) -> dict[str, float]:
    return _finite_blob(bench_field_linguistics(seed))


def bench_translation_theory_family(seed: int = _SEED + 50602) -> dict[str, float]:
    return _finite_blob(bench_translation_theory(seed))


def bench_sign_linguistics_family(seed: int = _SEED + 50603) -> dict[str, float]:
    return _finite_blob(bench_sign_linguistics(seed))


def bench_linguistic_typology_family(seed: int = _SEED + 50604) -> dict[str, float]:
    return _finite_blob(bench_linguistic_typology(seed))


def bench_language_acquisition_family(seed: int = _SEED + 50605) -> dict[str, float]:
    return _finite_blob(bench_language_acquisition(seed))
