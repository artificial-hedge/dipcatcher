"""Wave-1138 economics-5 canon tests."""

from __future__ import annotations

from quant_fund.models.behavioral_economics import bench_behavioral_economics
from quant_fund.models.econ_neuroscience import bench_econ_neuroscience
from quant_fund.models.evolutionary_economics import bench_evolutionary_economics
from quant_fund.models.experimental_economics_2 import bench_experimental_economics_2
from quant_fund.models.institutional_economics import bench_institutional_economics
from quant_fund.models.political_economy_2 import bench_political_economy_2


def test_behavioral_economics():
    assert bench_behavioral_economics()["synthetic_behavioral_economics"] == 1.0


def test_econ_neuroscience():
    assert bench_econ_neuroscience()["synthetic_econ_neuroscience"] == 1.0


def test_experimental_economics_2():
    assert bench_experimental_economics_2()["synthetic_experimental_economics_2"] == 1.0


def test_institutional_economics():
    assert bench_institutional_economics()["synthetic_institutional_economics"] == 1.0


def test_evolutionary_economics():
    assert bench_evolutionary_economics()["synthetic_evolutionary_economics"] == 1.0


def test_political_economy_2():
    assert bench_political_economy_2()["synthetic_political_economy_2"] == 1.0
