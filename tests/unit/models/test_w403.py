from quant_fund.models.hopf_invariant import bench_hopf_invariant
from quant_fund.models.j_hom_toy import bench_j_hom_toy
from quant_fund.models.pi_stems import bench_pi_stems
from quant_fund.models.spectral_atiyah import bench_spectral_atiyah
from quant_fund.models.thom_spectrum import bench_thom_spectrum
from quant_fund.models.toda_bracket import bench_toda_bracket


def test_j_hom_toy():
    assert bench_j_hom_toy()["synthetic_j_hom_toy"] == 1.0


def test_toda_bracket():
    assert bench_toda_bracket()["synthetic_toda_bracket"] == 1.0


def test_spectral_atiyah():
    assert bench_spectral_atiyah()["synthetic_spectral_atiyah"] == 1.0


def test_pi_stems():
    assert bench_pi_stems()["synthetic_pi_stems"] == 1.0


def test_hopf_invariant():
    assert bench_hopf_invariant()["synthetic_hopf_invariant"] == 1.0


def test_thom_spectrum():
    assert bench_thom_spectrum()["synthetic_thom_spectrum"] == 1.0
