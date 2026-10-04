"""Wave-1213 neurology-3 canon benchmark adapters (SYNTHETIC)."""

from __future__ import annotations

import math
from collections.abc import Iterable

from quant_fund.models.neurorehabilitation import bench_neurorehabilitation
from quant_fund.models.neurosurgery_studies import bench_neurosurgery_studies
from quant_fund.models.neurotoxicology import bench_neurotoxicology
from quant_fund.models.neurotrauma import bench_neurotrauma
from quant_fund.models.neurovascular_surgery import bench_neurovascular_surgery
from quant_fund.models.spinal_cord_medicine import bench_spinal_cord_medicine

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


def bench_neurosurgery_studies_family(seed: int = _SEED + 60100) -> dict[str, float]:
    return _finite_blob(bench_neurosurgery_studies(seed))


def bench_neurotrauma_family(seed: int = _SEED + 60101) -> dict[str, float]:
    return _finite_blob(bench_neurotrauma(seed))


def bench_neurotoxicology_family(seed: int = _SEED + 60102) -> dict[str, float]:
    return _finite_blob(bench_neurotoxicology(seed))


def bench_neurorehabilitation_family(seed: int = _SEED + 60103) -> dict[str, float]:
    return _finite_blob(bench_neurorehabilitation(seed))


def bench_neurovascular_surgery_family(seed: int = _SEED + 60104) -> dict[str, float]:
    return _finite_blob(bench_neurovascular_surgery(seed))


def bench_spinal_cord_medicine_family(seed: int = _SEED + 60105) -> dict[str, float]:
    return _finite_blob(bench_spinal_cord_medicine(seed))
