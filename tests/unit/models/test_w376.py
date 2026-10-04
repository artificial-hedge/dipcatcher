from quant_fund.models.cofibration import bench_cofibration
from quant_fund.models.fibration import bench_fibration
from quant_fund.models.serre_ss import bench_serre_ss
from quant_fund.models.spectra import bench_spectra
from quant_fund.models.suspension import bench_suspension
from quant_fund.models.whitehead import bench_whitehead


def test_fibration():
    assert bench_fibration()["synthetic_fibration"] == 1.0


def test_cofibration():
    assert bench_cofibration()["synthetic_cofibration"] == 1.0


def test_serre_ss():
    assert bench_serre_ss()["synthetic_serre_ss"] == 1.0


def test_whitehead():
    assert bench_whitehead()["synthetic_whitehead"] == 1.0


def test_suspension():
    assert bench_suspension()["synthetic_suspension"] == 1.0


def test_spectra():
    assert bench_spectra()["synthetic_spectra"] == 1.0
