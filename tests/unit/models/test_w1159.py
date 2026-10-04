"""Wave-1159 engineering canon tests."""

from __future__ import annotations

from quant_fund.models.aerospace_engineering_2 import bench_aerospace_engineering_2
from quant_fund.models.biomedical_engineering_2 import bench_biomedical_engineering_2
from quant_fund.models.chemical_engineering_2 import bench_chemical_engineering_2
from quant_fund.models.civil_engineering_2 import bench_civil_engineering_2
from quant_fund.models.electrical_engineering_2 import bench_electrical_engineering_2
from quant_fund.models.mechanical_engineering_2 import bench_mechanical_engineering_2


def test_biomedical_engineering_2():
    assert bench_biomedical_engineering_2()["synthetic_biomedical_engineering_2"] == 1.0


def test_chemical_engineering_2():
    assert bench_chemical_engineering_2()["synthetic_chemical_engineering_2"] == 1.0


def test_mechanical_engineering_2():
    assert bench_mechanical_engineering_2()["synthetic_mechanical_engineering_2"] == 1.0


def test_civil_engineering_2():
    assert bench_civil_engineering_2()["synthetic_civil_engineering_2"] == 1.0


def test_electrical_engineering_2():
    assert bench_electrical_engineering_2()["synthetic_electrical_engineering_2"] == 1.0


def test_aerospace_engineering_2():
    assert bench_aerospace_engineering_2()["synthetic_aerospace_engineering_2"] == 1.0
