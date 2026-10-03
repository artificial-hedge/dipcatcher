from quant_fund.models.ciucu_dimers import bench_ciucu_dimers
from quant_fund.models.cohn_elkies import bench_cohn_elkies
from quant_fund.models.durfee_arctic import bench_durfee_arctic
from quant_fund.models.karl_dimers import bench_karl_dimers
from quant_fund.models.kassel_kenyon import bench_kassel_kenyon
from quant_fund.models.petrov_dimer import bench_petrov_dimer


def test_kassel_kenyon():
    assert bench_kassel_kenyon()["synthetic_kassel_kenyon"] == 1.0


def test_ciucu_dimers():
    assert bench_ciucu_dimers()["synthetic_ciucu_dimers"] == 1.0


def test_karl_dimers():
    assert bench_karl_dimers()["synthetic_karl_dimers"] == 1.0


def test_petrov_dimer():
    assert bench_petrov_dimer()["synthetic_petrov_dimer"] == 1.0


def test_durfee_arctic():
    assert bench_durfee_arctic()["synthetic_durfee_arctic"] == 1.0


def test_cohn_elkies():
    assert bench_cohn_elkies()["synthetic_cohn_elkies"] == 1.0
