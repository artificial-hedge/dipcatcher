"""Tests for wave 1258 omics canon adapters (SYNTHETIC)."""

from __future__ import annotations

import pytest

from quant_fund.research import benches_w1258 as m


@pytest.mark.parametrize(
    "fn",
    [
        m.bench_transcriptome_studies_family,
        m.bench_proteome_studies_family,
        m.bench_metabolome_studies_family,
        m.bench_microbiome_studies_family,
        m.bench_methylome_studies_family,
        m.bench_interactome_studies_family,
    ],
)
def test_benches_w1258_return_unit_floats(fn) -> None:
    vals = fn()
    assert vals
    assert all(isinstance(v, float) and 0.0 <= v <= 1.0 for v in vals)
