"""Wave-1044 mining-engineering canon tests."""

from __future__ import annotations

from quant_fund.models.blasting_engineering import bench_blasting_engineering
from quant_fund.models.mine_design import bench_mine_design
from quant_fund.models.mine_ventilation import bench_mine_ventilation
from quant_fund.models.mineral_processing import bench_mineral_processing
from quant_fund.models.ore_reserve_estimation import bench_ore_reserve_estimation
from quant_fund.models.rock_mechanics import bench_rock_mechanics


def test_mine_design():
    assert bench_mine_design()["synthetic_mine_design"] == 1.0


def test_rock_mechanics():
    assert bench_rock_mechanics()["synthetic_rock_mechanics"] == 1.0


def test_mineral_processing():
    assert bench_mineral_processing()["synthetic_mineral_processing"] == 1.0


def test_blasting_engineering():
    assert bench_blasting_engineering()["synthetic_blasting_engineering"] == 1.0


def test_mine_ventilation():
    assert bench_mine_ventilation()["synthetic_mine_ventilation"] == 1.0


def test_ore_reserve_estimation():
    assert bench_ore_reserve_estimation()["synthetic_ore_reserve_estimation"] == 1.0
