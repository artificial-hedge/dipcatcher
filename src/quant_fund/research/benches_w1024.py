"""Wave-1024 computational-biology canon benchmark adapters (SYNTHETIC)."""

from __future__ import annotations

import math
from collections.abc import Iterable

from quant_fund.models.dna_sequencing import bench_dna_sequencing
from quant_fund.models.gene_expression import bench_gene_expression
from quant_fund.models.metabolomics import bench_metabolomics
from quant_fund.models.phylogenetics import bench_phylogenetics
from quant_fund.models.protein_folding import bench_protein_folding
from quant_fund.models.systems_biology import bench_systems_biology

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


def bench_protein_folding_family(seed: int = _SEED + 41200) -> dict[str, float]:
    return _finite_blob(bench_protein_folding(seed))


def bench_dna_sequencing_family(seed: int = _SEED + 41201) -> dict[str, float]:
    return _finite_blob(bench_dna_sequencing(seed))


def bench_phylogenetics_family(seed: int = _SEED + 41202) -> dict[str, float]:
    return _finite_blob(bench_phylogenetics(seed))


def bench_gene_expression_family(seed: int = _SEED + 41203) -> dict[str, float]:
    return _finite_blob(bench_gene_expression(seed))


def bench_metabolomics_family(seed: int = _SEED + 41204) -> dict[str, float]:
    return _finite_blob(bench_metabolomics(seed))


def bench_systems_biology_family(seed: int = _SEED + 41205) -> dict[str, float]:
    return _finite_blob(bench_systems_biology(seed))
