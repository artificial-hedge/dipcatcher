from quant_fund.models.adams_diff import bench_adams_diff
from quant_fund.models.cartan_eilenberg import bench_cartan_eilenberg
from quant_fund.models.deriv_hom import bench_deriv_hom
from quant_fund.models.groth_spectral import bench_groth_spectral
from quant_fund.models.hypercohom import bench_hypercohom
from quant_fund.models.serre_ss2 import bench_serre_ss2


def test_groth_spectral():
    assert bench_groth_spectral()["synthetic_groth_spectral"] == 1.0


def test_serre_ss2():
    assert bench_serre_ss2()["synthetic_serre_ss2"] == 1.0


def test_hypercohom():
    assert bench_hypercohom()["synthetic_hypercohom"] == 1.0


def test_deriv_hom():
    assert bench_deriv_hom()["synthetic_deriv_hom"] == 1.0


def test_cartan_eilenberg():
    assert bench_cartan_eilenberg()["synthetic_cartan_eilenberg"] == 1.0


def test_adams_diff():
    assert bench_adams_diff()["synthetic_adams_diff"] == 1.0
