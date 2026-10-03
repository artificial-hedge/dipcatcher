from quant_fund.models.chain_complex import bench_chain_complex
from quant_fund.models.hilbert_series import bench_hilbert_series
from quant_fund.models.sheaf_check import bench_sheaf_check
from quant_fund.models.snake_lemma import bench_snake_lemma
from quant_fund.models.tor_ext import bench_tor_ext
from quant_fund.models.variety_morph import bench_variety_morph


def test_chain_complex():
    assert bench_chain_complex()["synthetic_chain_complex"] == 1.0


def test_tor_ext():
    assert bench_tor_ext()["synthetic_tor_ext"] == 1.0


def test_sheaf_check():
    assert bench_sheaf_check()["synthetic_sheaf_check"] == 1.0


def test_hilbert_series():
    assert bench_hilbert_series()["synthetic_hilbert_series"] == 1.0


def test_snake_lemma():
    assert bench_snake_lemma()["synthetic_snake_lemma"] == 1.0


def test_variety_morph():
    assert bench_variety_morph()["synthetic_variety_morph"] == 1.0
