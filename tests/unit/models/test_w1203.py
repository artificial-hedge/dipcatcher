"""Wave-1203 molecular-medicine canon tests."""

from __future__ import annotations

from quant_fund.models.genomic_medicine import bench_genomic_medicine
from quant_fund.models.laboratory_medicine import bench_laboratory_medicine
from quant_fund.models.molecular_diagnostics import bench_molecular_diagnostics
from quant_fund.models.precision_medicine import bench_precision_medicine
from quant_fund.models.travel_medicine import bench_travel_medicine
from quant_fund.models.tropical_medicine import bench_tropical_medicine


def test_tropical_medicine():
    assert bench_tropical_medicine()["synthetic_tropical_medicine"] == 1.0


def test_travel_medicine():
    assert bench_travel_medicine()["synthetic_travel_medicine"] == 1.0


def test_genomic_medicine():
    assert bench_genomic_medicine()["synthetic_genomic_medicine"] == 1.0


def test_precision_medicine():
    assert bench_precision_medicine()["synthetic_precision_medicine"] == 1.0


def test_molecular_diagnostics():
    assert bench_molecular_diagnostics()["synthetic_molecular_diagnostics"] == 1.0


def test_laboratory_medicine():
    assert bench_laboratory_medicine()["synthetic_laboratory_medicine"] == 1.0
