"""Wave-961 semigroup-theory canon tests."""

from __future__ import annotations

from quant_fund.models.analytic_semigroup import bench_analytic_semigroup
from quant_fund.models.c0_semigroup import bench_c0_semigroup
from quant_fund.models.cosine_family import bench_cosine_family
from quant_fund.models.hille_yosida import bench_hille_yosida
from quant_fund.models.lumer_phillips import bench_lumer_phillips
from quant_fund.models.trotter_kato import bench_trotter_kato


def test_c0_semigroup():
    assert bench_c0_semigroup()["synthetic_c0_semigroup"] == 1.0


def test_hille_yosida():
    assert bench_hille_yosida()["synthetic_hille_yosida"] == 1.0


def test_lumer_phillips():
    assert bench_lumer_phillips()["synthetic_lumer_phillips"] == 1.0


def test_analytic_semigroup():
    assert bench_analytic_semigroup()["synthetic_analytic_semigroup"] == 1.0


def test_cosine_family():
    assert bench_cosine_family()["synthetic_cosine_family"] == 1.0


def test_trotter_kato():
    assert bench_trotter_kato()["synthetic_trotter_kato"] == 1.0
