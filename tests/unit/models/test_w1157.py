"""Wave-1157 computing-sciences canon tests."""

from __future__ import annotations

from quant_fund.models.artificial_intelligence import bench_artificial_intelligence
from quant_fund.models.computer_science_2 import bench_computer_science_2
from quant_fund.models.data_engineering import bench_data_engineering
from quant_fund.models.information_theory_2 import bench_information_theory_2
from quant_fund.models.machine_learning_2 import bench_machine_learning_2
from quant_fund.models.software_engineering import bench_software_engineering


def test_computer_science_2():
    assert bench_computer_science_2()["synthetic_computer_science_2"] == 1.0


def test_software_engineering():
    assert bench_software_engineering()["synthetic_software_engineering"] == 1.0


def test_machine_learning_2():
    assert bench_machine_learning_2()["synthetic_machine_learning_2"] == 1.0


def test_artificial_intelligence():
    assert bench_artificial_intelligence()["synthetic_artificial_intelligence"] == 1.0


def test_data_engineering():
    assert bench_data_engineering()["synthetic_data_engineering"] == 1.0


def test_information_theory_2():
    assert bench_information_theory_2()["synthetic_information_theory_2"] == 1.0
