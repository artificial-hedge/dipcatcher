"""Wave-1212 neurology-2 canon benchmark adapters (SYNTHETIC)."""

from __future__ import annotations

import math
from collections.abc import Iterable

from quant_fund.models.neuro_ophthalmology import bench_neuro_ophthalmology
from quant_fund.models.neurocritical_care import bench_neurocritical_care
from quant_fund.models.neurogenetics import bench_neurogenetics
from quant_fund.models.neuroimmunology import bench_neuroimmunology
from quant_fund.models.neuromuscular_medicine import bench_neuromuscular_medicine
from quant_fund.models.neurovascular_studies import bench_neurovascular_studies

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


def bench_neurocritical_care_family(seed: int = _SEED + 60000) -> dict[str, float]:
    return _finite_blob(bench_neurocritical_care(seed))


def bench_neurovascular_studies_family(seed: int = _SEED + 60001) -> dict[str, float]:
    return _finite_blob(bench_neurovascular_studies(seed))


def bench_neuromuscular_medicine_family(seed: int = _SEED + 60002) -> dict[str, float]:
    return _finite_blob(bench_neuromuscular_medicine(seed))


def bench_neuro_ophthalmology_family(seed: int = _SEED + 60003) -> dict[str, float]:
    return _finite_blob(bench_neuro_ophthalmology(seed))


def bench_neuroimmunology_family(seed: int = _SEED + 60004) -> dict[str, float]:
    return _finite_blob(bench_neuroimmunology(seed))


def bench_neurogenetics_family(seed: int = _SEED + 60005) -> dict[str, float]:
    return _finite_blob(bench_neurogenetics(seed))
