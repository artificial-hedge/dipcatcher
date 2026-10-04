"""Wave-1163 business canon tests."""

from __future__ import annotations

from quant_fund.models.accounting_2 import bench_accounting_2
from quant_fund.models.business_administration import bench_business_administration
from quant_fund.models.entrepreneurship_2 import bench_entrepreneurship_2
from quant_fund.models.finance_4 import bench_finance_4
from quant_fund.models.management_2 import bench_management_2
from quant_fund.models.marketing_2 import bench_marketing_2


def test_accounting_2():
    assert bench_accounting_2()["synthetic_accounting_2"] == 1.0


def test_finance_4():
    assert bench_finance_4()["synthetic_finance_4"] == 1.0


def test_marketing_2():
    assert bench_marketing_2()["synthetic_marketing_2"] == 1.0


def test_management_2():
    assert bench_management_2()["synthetic_management_2"] == 1.0


def test_entrepreneurship_2():
    assert bench_entrepreneurship_2()["synthetic_entrepreneurship_2"] == 1.0


def test_business_administration():
    assert bench_business_administration()["synthetic_business_administration"] == 1.0
