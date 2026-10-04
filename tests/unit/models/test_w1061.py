"""Wave-1061 law canon tests."""

from __future__ import annotations

from quant_fund.models.administrative_law import bench_administrative_law
from quant_fund.models.constitutional_law import bench_constitutional_law
from quant_fund.models.contract_law import bench_contract_law
from quant_fund.models.criminal_law import bench_criminal_law
from quant_fund.models.international_law import bench_international_law
from quant_fund.models.tort_law import bench_tort_law


def test_constitutional_law():
    assert bench_constitutional_law()["synthetic_constitutional_law"] == 1.0


def test_criminal_law():
    assert bench_criminal_law()["synthetic_criminal_law"] == 1.0


def test_contract_law():
    assert bench_contract_law()["synthetic_contract_law"] == 1.0


def test_tort_law():
    assert bench_tort_law()["synthetic_tort_law"] == 1.0


def test_administrative_law():
    assert bench_administrative_law()["synthetic_administrative_law"] == 1.0


def test_international_law():
    assert bench_international_law()["synthetic_international_law"] == 1.0
