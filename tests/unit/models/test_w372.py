from quant_fund.models.girsanov import bench_girsanov
from quant_fund.models.ito_lemma import bench_ito_lemma
from quant_fund.models.local_time import bench_local_time
from quant_fund.models.malliavin import bench_malliavin
from quant_fund.models.quadratic_var import bench_quadratic_var
from quant_fund.models.sde_strong import bench_sde_strong


def test_ito_lemma():
    assert bench_ito_lemma()["synthetic_ito_lemma"] == 1.0


def test_girsanov():
    assert bench_girsanov()["synthetic_girsanov"] == 1.0


def test_sde_strong():
    assert bench_sde_strong()["synthetic_sde_strong"] == 1.0


def test_local_time():
    assert bench_local_time()["synthetic_local_time"] == 1.0


def test_quadratic_var():
    assert bench_quadratic_var()["synthetic_quadratic_var"] == 1.0


def test_malliavin():
    assert bench_malliavin()["synthetic_malliavin"] == 1.0
