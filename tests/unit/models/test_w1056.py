"""Wave-1056 political-science canon tests."""

from __future__ import annotations

from quant_fund.models.comparative_politics import bench_comparative_politics
from quant_fund.models.electoral_systems import bench_electoral_systems
from quant_fund.models.international_relations import bench_international_relations
from quant_fund.models.political_economy import bench_political_economy
from quant_fund.models.political_theory import bench_political_theory
from quant_fund.models.public_administration import bench_public_administration


def test_comparative_politics():
    assert bench_comparative_politics()["synthetic_comparative_politics"] == 1.0


def test_international_relations():
    assert bench_international_relations()["synthetic_international_relations"] == 1.0


def test_political_theory():
    assert bench_political_theory()["synthetic_political_theory"] == 1.0


def test_public_administration():
    assert bench_public_administration()["synthetic_public_administration"] == 1.0


def test_political_economy():
    assert bench_political_economy()["synthetic_political_economy"] == 1.0


def test_electoral_systems():
    assert bench_electoral_systems()["synthetic_electoral_systems"] == 1.0
