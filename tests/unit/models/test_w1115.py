"""Wave-1115 economics-3 canon tests."""

from __future__ import annotations

from quant_fund.models.financial_economics import bench_financial_economics
from quant_fund.models.industrial_organization import bench_industrial_organization
from quant_fund.models.international_economics import bench_international_economics
from quant_fund.models.labor_economics import bench_labor_economics
from quant_fund.models.monetary_economics import bench_monetary_economics
from quant_fund.models.public_economics import bench_public_economics


def test_labor_economics():
    assert bench_labor_economics()["synthetic_labor_economics"] == 1.0


def test_public_economics():
    assert bench_public_economics()["synthetic_public_economics"] == 1.0


def test_industrial_organization():
    assert bench_industrial_organization()["synthetic_industrial_organization"] == 1.0


def test_international_economics():
    assert bench_international_economics()["synthetic_international_economics"] == 1.0


def test_financial_economics():
    assert bench_financial_economics()["synthetic_financial_economics"] == 1.0


def test_monetary_economics():
    assert bench_monetary_economics()["synthetic_monetary_economics"] == 1.0
