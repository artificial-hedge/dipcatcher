from quant_fund.models.adjunction2 import bench_adjunction2
from quant_fund.models.cech_cohom import bench_cech_cohom
from quant_fund.models.flattening import bench_flattening
from quant_fund.models.hilbert_scheme import bench_hilbert_scheme
from quant_fund.models.scheme_fiber import bench_scheme_fiber
from quant_fund.models.serre_duality import bench_serre_duality


def test_cech_cohom():
    assert bench_cech_cohom()["synthetic_cech_cohom"] == 1.0


def test_serre_duality():
    assert bench_serre_duality()["synthetic_serre_duality"] == 1.0


def test_adjunction2():
    assert bench_adjunction2()["synthetic_adjunction2"] == 1.0


def test_scheme_fiber():
    assert bench_scheme_fiber()["synthetic_scheme_fiber"] == 1.0


def test_hilbert_scheme():
    assert bench_hilbert_scheme()["synthetic_hilbert_scheme"] == 1.0


def test_flattening():
    assert bench_flattening()["synthetic_flattening"] == 1.0
