"""Wave-1024 computational-biology canon tests."""

from __future__ import annotations

from quant_fund.models.dna_sequencing import bench_dna_sequencing
from quant_fund.models.gene_expression import bench_gene_expression
from quant_fund.models.metabolomics import bench_metabolomics
from quant_fund.models.phylogenetics import bench_phylogenetics
from quant_fund.models.protein_folding import bench_protein_folding
from quant_fund.models.systems_biology import bench_systems_biology


def test_protein_folding():
    assert bench_protein_folding()["synthetic_protein_folding"] == 1.0


def test_dna_sequencing():
    assert bench_dna_sequencing()["synthetic_dna_sequencing"] == 1.0


def test_phylogenetics():
    assert bench_phylogenetics()["synthetic_phylogenetics"] == 1.0


def test_gene_expression():
    assert bench_gene_expression()["synthetic_gene_expression"] == 1.0


def test_metabolomics():
    assert bench_metabolomics()["synthetic_metabolomics"] == 1.0


def test_systems_biology():
    assert bench_systems_biology()["synthetic_systems_biology"] == 1.0
