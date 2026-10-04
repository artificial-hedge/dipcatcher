from quant_fund.models.homotopy_coherent import bench_homotopy_coherent
from quant_fund.models.htc_colimit import bench_htc_colimit
from quant_fund.models.joyal_model import bench_joyal_model
from quant_fund.models.marking_qcat import bench_marking_qcat
from quant_fund.models.nerve_quasi import bench_nerve_quasi
from quant_fund.models.quasi_cat import bench_quasi_cat


def test_quasi_cat():
    assert bench_quasi_cat()["synthetic_quasi_cat"] == 1.0


def test_joyal_model():
    assert bench_joyal_model()["synthetic_joyal_model"] == 1.0


def test_homotopy_coherent():
    assert bench_homotopy_coherent()["synthetic_homotopy_coherent"] == 1.0


def test_nerve_quasi():
    assert bench_nerve_quasi()["synthetic_nerve_quasi"] == 1.0


def test_htc_colimit():
    assert bench_htc_colimit()["synthetic_htc_colimit"] == 1.0


def test_marking_qcat():
    assert bench_marking_qcat()["synthetic_marking_qcat"] == 1.0
