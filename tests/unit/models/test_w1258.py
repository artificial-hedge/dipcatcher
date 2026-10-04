"""Tests for wave 1258 omics canon families (SYNTHETIC)."""
from __future__ import annotations

import pytest

import quant_fund.models.interactome_studies as interactome_studies
import quant_fund.models.metabolome_studies as metabolome_studies
import quant_fund.models.methylome_studies as methylome_studies
import quant_fund.models.microbiome_studies as microbiome_studies
import quant_fund.models.proteome_studies as proteome_studies
import quant_fund.models.transcriptome_studies as transcriptome_studies


@pytest.mark.parametrize("name", ['transcriptome_studies', 'proteome_studies', 'metabolome_studies', 'microbiome_studies', 'methylome_studies', 'interactome_studies'])
def test_w1258_ok_aux_contract(name: str) -> None:
    mod = {"transcriptome_studies": transcriptome_studies, "proteome_studies": proteome_studies, "metabolome_studies": metabolome_studies, "microbiome_studies": microbiome_studies, "methylome_studies": methylome_studies, "interactome_studies": interactome_studies}[name]
    ok = getattr(mod, f"{name}_ok")
    aux = getattr(mod, f"{name}_aux")
    assert ok(True, True) is True
    assert ok(False, True) is False
    assert aux(True) is True
    assert aux(False) is False
