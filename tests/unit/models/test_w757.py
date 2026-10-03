from quant_fund.models.camia_newman import bench_camia_newman
from quant_fund.models.dubedat_cle import bench_dubedat_cle
from quant_fund.models.kemppainen_werner import (
    bench_kemppainen_werner,
)
from quant_fund.models.miller_watson_cle import (
    bench_miller_watson_cle,
)
from quant_fund.models.rivera_cle import bench_rivera_cle
from quant_fund.models.sheffield_werner_cle import (
    bench_sheffield_werner_cle,
)


def test_sheffield_werner_cle():
    assert bench_sheffield_werner_cle()["synthetic_sheffield_werner_cle"] == 1.0


def test_miller_watson_cle():
    assert bench_miller_watson_cle()["synthetic_miller_watson_cle"] == 1.0


def test_camia_newman():
    assert bench_camia_newman()["synthetic_camia_newman"] == 1.0


def test_dubedat_cle():
    assert bench_dubedat_cle()["synthetic_dubedat_cle"] == 1.0


def test_kemppainen_werner():
    assert bench_kemppainen_werner()["synthetic_kemppainen_werner"] == 1.0


def test_rivera_cle():
    assert bench_rivera_cle()["synthetic_rivera_cle"] == 1.0
