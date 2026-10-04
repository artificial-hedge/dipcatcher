from quant_fund.models.derived_functor import bench_derived_functor
from quant_fund.models.ext_compute import bench_ext_compute
from quant_fund.models.koszul_homology import bench_koszul_homology
from quant_fund.models.mapping_degree import bench_mapping_degree
from quant_fund.models.spectral_seq import bench_spectral_seq
from quant_fund.models.tor_compute import bench_tor_compute


def test_derived_functor():
    assert bench_derived_functor()["synthetic_derived_functor"] == 1.0


def test_ext_compute():
    assert bench_ext_compute()["synthetic_ext_compute"] == 1.0


def test_tor_compute():
    assert bench_tor_compute()["synthetic_tor_compute"] == 1.0


def test_spectral_seq():
    assert bench_spectral_seq()["synthetic_spectral_seq"] == 1.0


def test_koszul_homology():
    assert bench_koszul_homology()["synthetic_koszul_homology"] == 1.0


def test_mapping_degree():
    assert bench_mapping_degree()["synthetic_mapping_degree"] == 1.0
