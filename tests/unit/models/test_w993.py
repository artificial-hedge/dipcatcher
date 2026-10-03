"""Wave-993 nonlinear-functional-analysis canon tests."""

from __future__ import annotations

from quant_fund.models.degree_theory import bench_degree_theory
from quant_fund.models.krein_rutman import bench_krein_rutman
from quant_fund.models.maximal_monotone import bench_maximal_monotone
from quant_fund.models.minty_browder import bench_minty_browder
from quant_fund.models.monotone_op import bench_monotone_op
from quant_fund.models.schauder_fixed import bench_schauder_fixed


def test_monotone_op():
    assert bench_monotone_op()["synthetic_monotone_op"] == 1.0


def test_degree_theory():
    assert bench_degree_theory()["synthetic_degree_theory"] == 1.0


def test_schauder_fixed():
    assert bench_schauder_fixed()["synthetic_schauder_fixed"] == 1.0


def test_krein_rutman():
    assert bench_krein_rutman()["synthetic_krein_rutman"] == 1.0


def test_minty_browder():
    assert bench_minty_browder()["synthetic_minty_browder"] == 1.0


def test_maximal_monotone():
    assert bench_maximal_monotone()["synthetic_maximal_monotone"] == 1.0
