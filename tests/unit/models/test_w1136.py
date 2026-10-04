"""Wave-1136 law-4 canon tests."""

from __future__ import annotations

from quant_fund.models.environmental_law import bench_environmental_law
from quant_fund.models.evidence_law import bench_evidence_law
from quant_fund.models.family_law import bench_family_law
from quant_fund.models.immigration_law import bench_immigration_law
from quant_fund.models.labor_law import bench_labor_law
from quant_fund.models.tax_law import bench_tax_law


def test_environmental_law():
    assert bench_environmental_law()["synthetic_environmental_law"] == 1.0


def test_family_law():
    assert bench_family_law()["synthetic_family_law"] == 1.0


def test_labor_law():
    assert bench_labor_law()["synthetic_labor_law"] == 1.0


def test_tax_law():
    assert bench_tax_law()["synthetic_tax_law"] == 1.0


def test_evidence_law():
    assert bench_evidence_law()["synthetic_evidence_law"] == 1.0


def test_immigration_law():
    assert bench_immigration_law()["synthetic_immigration_law"] == 1.0
