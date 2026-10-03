from quant_fund.models.cohend import bench_cohend
from quant_fund.models.dold_kan import bench_dold_kan
from quant_fund.models.eilenberg_zilber import bench_eilenberg_zilber
from quant_fund.models.postnikov import bench_postnikov
from quant_fund.models.spectral_seq2 import bench_spectral_seq2
from quant_fund.models.stable_range import bench_stable_range


def test_spectral_seq2():
    assert bench_spectral_seq2()["synthetic_spectral_seq2"] == 1.0


def test_eilenberg_zilber():
    assert bench_eilenberg_zilber()["synthetic_eilenberg_zilber"] == 1.0


def test_dold_kan():
    assert bench_dold_kan()["synthetic_dold_kan"] == 1.0


def test_postnikov():
    assert bench_postnikov()["synthetic_postnikov"] == 1.0


def test_stable_range():
    assert bench_stable_range()["synthetic_stable_range"] == 1.0


def test_cohend():
    assert bench_cohend()["synthetic_cohend"] == 1.0
