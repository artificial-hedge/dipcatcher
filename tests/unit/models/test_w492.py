from quant_fund.models.kato_fontaine import bench_kato_fontaine
from quant_fund.models.log_crystalline import bench_log_crystalline
from quant_fund.models.log_derham import bench_log_derham
from quant_fund.models.log_etale import bench_log_etale
from quant_fund.models.log_smooth import bench_log_smooth
from quant_fund.models.log_structure import bench_log_structure


def test_log_structure():
    assert bench_log_structure()["synthetic_log_structure"] == 1.0


def test_kato_fontaine():
    assert bench_kato_fontaine()["synthetic_kato_fontaine"] == 1.0


def test_log_smooth():
    assert bench_log_smooth()["synthetic_log_smooth"] == 1.0


def test_log_etale():
    assert bench_log_etale()["synthetic_log_etale"] == 1.0


def test_log_derham():
    assert bench_log_derham()["synthetic_log_derham"] == 1.0


def test_log_crystalline():
    assert bench_log_crystalline()["synthetic_log_crystalline"] == 1.0
