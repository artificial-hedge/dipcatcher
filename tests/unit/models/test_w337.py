from quant_fund.models.church_encoding import bench_church_encoding
from quant_fund.models.de_bruijn import bench_de_bruijn
from quant_fund.models.knuth_bendix import bench_knuth_bendix
from quant_fund.models.lambda_typing import bench_lambda_typing
from quant_fund.models.ski_combinator import bench_ski_combinator
from quant_fund.models.unification import bench_unification


def test_ski_combinator():
    assert bench_ski_combinator()["synthetic_ski_combinator"] == 1.0


def test_de_bruijn():
    assert bench_de_bruijn()["synthetic_de_bruijn"] == 1.0


def test_church_encoding():
    assert bench_church_encoding()["synthetic_church_encoding"] == 1.0


def test_lambda_typing():
    assert bench_lambda_typing()["synthetic_lambda_typing"] == 1.0


def test_unification():
    assert bench_unification()["synthetic_unification"] == 1.0


def test_knuth_bendix():
    assert bench_knuth_bendix()["synthetic_knuth_bendix"] == 1.0
