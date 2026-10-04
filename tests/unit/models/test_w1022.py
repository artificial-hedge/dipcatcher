"""Wave-1022 economics canon tests."""

from __future__ import annotations

from quant_fund.models.auction_theory2 import bench_auction_theory2
from quant_fund.models.growth_theory import bench_growth_theory
from quant_fund.models.mechanism_design import bench_mechanism_design
from quant_fund.models.overlapping_gens import bench_overlapping_gens
from quant_fund.models.real_business import bench_real_business
from quant_fund.models.search_matching import bench_search_matching


def test_growth_theory():
    assert bench_growth_theory()["synthetic_growth_theory"] == 1.0


def test_overlapping_gens():
    assert bench_overlapping_gens()["synthetic_overlapping_gens"] == 1.0


def test_real_business():
    assert bench_real_business()["synthetic_real_business"] == 1.0


def test_search_matching():
    assert bench_search_matching()["synthetic_search_matching"] == 1.0


def test_mechanism_design():
    assert bench_mechanism_design()["synthetic_mechanism_design"] == 1.0


def test_auction_theory2():
    assert bench_auction_theory2()["synthetic_auction_theory2"] == 1.0
