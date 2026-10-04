"""Wave-1172 business canon tests."""

from __future__ import annotations

from quant_fund.models.accounting_3 import bench_accounting_3
from quant_fund.models.entrepreneurship_3 import bench_entrepreneurship_3
from quant_fund.models.finance_5 import bench_finance_5
from quant_fund.models.management_3 import bench_management_3
from quant_fund.models.marketing_3 import bench_marketing_3
from quant_fund.models.organizational_behavior import bench_organizational_behavior


def test_management_3():
    assert bench_management_3()["synthetic_management_3"] == 1.0


def test_marketing_3():
    assert bench_marketing_3()["synthetic_marketing_3"] == 1.0


def test_accounting_3():
    assert bench_accounting_3()["synthetic_accounting_3"] == 1.0


def test_finance_5():
    assert bench_finance_5()["synthetic_finance_5"] == 1.0


def test_entrepreneurship_3():
    assert bench_entrepreneurship_3()["synthetic_entrepreneurship_3"] == 1.0


def test_organizational_behavior():
    assert bench_organizational_behavior()["synthetic_organizational_behavior"] == 1.0
