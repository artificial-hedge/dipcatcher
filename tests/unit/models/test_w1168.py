"""Wave-1168 logistics canon tests."""

from __future__ import annotations

from quant_fund.models.aviation_2 import bench_aviation_2
from quant_fund.models.logistics_2 import bench_logistics_2
from quant_fund.models.maritime_studies_2 import bench_maritime_studies_2
from quant_fund.models.supply_chain_2 import bench_supply_chain_2
from quant_fund.models.transportation_2 import bench_transportation_2
from quant_fund.models.warehousing_2 import bench_warehousing_2


def test_transportation_2():
    assert bench_transportation_2()["synthetic_transportation_2"] == 1.0


def test_logistics_2():
    assert bench_logistics_2()["synthetic_logistics_2"] == 1.0


def test_supply_chain_2():
    assert bench_supply_chain_2()["synthetic_supply_chain_2"] == 1.0


def test_warehousing_2():
    assert bench_warehousing_2()["synthetic_warehousing_2"] == 1.0


def test_maritime_studies_2():
    assert bench_maritime_studies_2()["synthetic_maritime_studies_2"] == 1.0


def test_aviation_2():
    assert bench_aviation_2()["synthetic_aviation_2"] == 1.0
