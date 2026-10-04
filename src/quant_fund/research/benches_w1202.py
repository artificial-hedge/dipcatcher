"""Wave-1202 infectious-disease canon benchmark adapters (SYNTHETIC)."""

from __future__ import annotations

import math
from collections.abc import Iterable

from quant_fund.models.entomology_medical import bench_entomology_medical
from quant_fund.models.medical_microbiology import bench_medical_microbiology
from quant_fund.models.mycology_studies import bench_mycology_studies
from quant_fund.models.parasitology_studies import bench_parasitology_studies
from quant_fund.models.public_health_microbiology import bench_public_health_microbiology
from quant_fund.models.vector_borne_diseases import bench_vector_borne_diseases

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


def bench_public_health_microbiology_family(seed: int = _SEED + 59000) -> dict[str, float]:
    return _finite_blob(bench_public_health_microbiology(seed))


def bench_medical_microbiology_family(seed: int = _SEED + 59001) -> dict[str, float]:
    return _finite_blob(bench_medical_microbiology(seed))


def bench_parasitology_studies_family(seed: int = _SEED + 59002) -> dict[str, float]:
    return _finite_blob(bench_parasitology_studies(seed))


def bench_mycology_studies_family(seed: int = _SEED + 59003) -> dict[str, float]:
    return _finite_blob(bench_mycology_studies(seed))


def bench_entomology_medical_family(seed: int = _SEED + 59004) -> dict[str, float]:
    return _finite_blob(bench_entomology_medical(seed))


def bench_vector_borne_diseases_family(seed: int = _SEED + 59005) -> dict[str, float]:
    return _finite_blob(bench_vector_borne_diseases(seed))
