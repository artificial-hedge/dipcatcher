"""Wave-1187 fashion canon tests."""

from __future__ import annotations

from quant_fund.models.apparel_studies import bench_apparel_studies
from quant_fund.models.costume_design import bench_costume_design
from quant_fund.models.fashion_studies import bench_fashion_studies
from quant_fund.models.footwear_design import bench_footwear_design
from quant_fund.models.jewelry_design import bench_jewelry_design
from quant_fund.models.textile_studies import bench_textile_studies


def test_fashion_studies():
    assert bench_fashion_studies()["synthetic_fashion_studies"] == 1.0


def test_textile_studies():
    assert bench_textile_studies()["synthetic_textile_studies"] == 1.0


def test_costume_design():
    assert bench_costume_design()["synthetic_costume_design"] == 1.0


def test_jewelry_design():
    assert bench_jewelry_design()["synthetic_jewelry_design"] == 1.0


def test_footwear_design():
    assert bench_footwear_design()["synthetic_footwear_design"] == 1.0


def test_apparel_studies():
    assert bench_apparel_studies()["synthetic_apparel_studies"] == 1.0
