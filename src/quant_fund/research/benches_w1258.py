"""Wave 1258 omics canon adapters (SYNTHETIC)."""
from __future__ import annotations

from typing import Callable

from quant_fund.models.transcriptome_studies import bench_transcriptome_studies
from quant_fund.models.proteome_studies import bench_proteome_studies
from quant_fund.models.metabolome_studies import bench_metabolome_studies
from quant_fund.models.microbiome_studies import bench_microbiome_studies
from quant_fund.models.methylome_studies import bench_methylome_studies
from quant_fund.models.interactome_studies import bench_interactome_studies

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 20261231


def _finite_blob(blob: dict[str, float]) -> dict[str, float]:
    assert isinstance(blob, dict)
    for key, value in blob.items():
        assert key not in _FORBIDDEN
        assert key.startswith("synthetic_")
        assert isinstance(value, float) and 0.0 <= value <= 1.0
    return blob


def _floats(blob: dict[str, float]) -> list[float]:
    return sorted(_finite_blob(blob).values())



def bench_transcriptome_studies_family(seed: int = _SEED + 64600) -> list[float]:
    return _floats(bench_transcriptome_studies(seed=seed))


def bench_proteome_studies_family(seed: int = _SEED + 64601) -> list[float]:
    return _floats(bench_proteome_studies(seed=seed))


def bench_metabolome_studies_family(seed: int = _SEED + 64602) -> list[float]:
    return _floats(bench_metabolome_studies(seed=seed))


def bench_microbiome_studies_family(seed: int = _SEED + 64603) -> list[float]:
    return _floats(bench_microbiome_studies(seed=seed))


def bench_methylome_studies_family(seed: int = _SEED + 64604) -> list[float]:
    return _floats(bench_methylome_studies(seed=seed))


def bench_interactome_studies_family(seed: int = _SEED + 64605) -> list[float]:
    return _floats(bench_interactome_studies(seed=seed))

