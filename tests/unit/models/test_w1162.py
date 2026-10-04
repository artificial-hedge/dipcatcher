"""Wave-1162 governance canon tests."""

from __future__ import annotations

from quant_fund.models.criminology_2 import bench_criminology_2
from quant_fund.models.international_relations_2 import bench_international_relations_2
from quant_fund.models.law_5 import bench_law_5
from quant_fund.models.military_science_2 import bench_military_science_2
from quant_fund.models.political_science_4 import bench_political_science_4
from quant_fund.models.public_administration_2 import bench_public_administration_2


def test_law_5():
    assert bench_law_5()["synthetic_law_5"] == 1.0


def test_political_science_4():
    assert bench_political_science_4()["synthetic_political_science_4"] == 1.0


def test_public_administration_2():
    assert bench_public_administration_2()["synthetic_public_administration_2"] == 1.0


def test_international_relations_2():
    assert bench_international_relations_2()["synthetic_international_relations_2"] == 1.0


def test_criminology_2():
    assert bench_criminology_2()["synthetic_criminology_2"] == 1.0


def test_military_science_2():
    assert bench_military_science_2()["synthetic_military_science_2"] == 1.0
