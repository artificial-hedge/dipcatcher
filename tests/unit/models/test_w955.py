"""Wave-955 linear-systems canon tests."""

from __future__ import annotations

from quant_fund.models.back_substitution import bench_back_substitution
from quant_fund.models.forward_substitution import bench_forward_substitution
from quant_fund.models.givens_rotation import bench_givens_rotation
from quant_fund.models.gram_determinant import bench_gram_determinant
from quant_fund.models.gram_matrix import bench_gram_matrix
from quant_fund.models.householder_reflect import bench_householder_reflect


def test_gram_matrix():
    assert bench_gram_matrix()["synthetic_gram_matrix"] == 1.0


def test_gram_determinant():
    assert bench_gram_determinant()["synthetic_gram_determinant"] == 1.0


def test_householder_reflect():
    assert bench_householder_reflect()["synthetic_householder_reflect"] == 1.0


def test_givens_rotation():
    assert bench_givens_rotation()["synthetic_givens_rotation"] == 1.0


def test_back_substitution():
    assert bench_back_substitution()["synthetic_back_substitution"] == 1.0


def test_forward_substitution():
    assert bench_forward_substitution()["synthetic_forward_substitution"] == 1.0
