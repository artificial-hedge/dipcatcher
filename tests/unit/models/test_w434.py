from quant_fund.models.automorphic_rep import (
    bench_automorphic_rep,
)
from quant_fund.models.eisenstein_srs import bench_eisenstein_srs
from quant_fund.models.fourier_coeff import bench_fourier_coeff
from quant_fund.models.hecke_operator import bench_hecke_operator
from quant_fund.models.langlands_dual import bench_langlands_dual
from quant_fund.models.satake_iso import bench_satake_iso


def test_satake_iso():
    assert bench_satake_iso()["synthetic_satake_iso"] == 1.0


def test_hecke_operator():
    assert bench_hecke_operator()["synthetic_hecke_operator"] == 1.0


def test_langlands_dual():
    assert bench_langlands_dual()["synthetic_langlands_dual"] == 1.0


def test_eisenstein_srs():
    assert bench_eisenstein_srs()["synthetic_eisenstein_srs"] == 1.0


def test_automorphic_rep():
    assert bench_automorphic_rep()["synthetic_automorphic_rep"] == 1.0


def test_fourier_coeff():
    assert bench_fourier_coeff()["synthetic_fourier_coeff"] == 1.0
