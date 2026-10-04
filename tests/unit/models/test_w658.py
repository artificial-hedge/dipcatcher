from quant_fund.models.beilinson_regulator import bench_beilinson_regulator
from quant_fund.models.f_motive import bench_f_motive
from quant_fund.models.hodge_motive import bench_hodge_motive
from quant_fund.models.motivic_galois import bench_motivic_galois
from quant_fund.models.period_realization import bench_period_realization
from quant_fund.models.tannakian_motive import bench_tannakian_motive


def test_motivic_galois():
    assert bench_motivic_galois()["synthetic_motivic_galois"] == 1.0


def test_tannakian_motive():
    assert bench_tannakian_motive()["synthetic_tannakian_motive"] == 1.0


def test_period_realization():
    assert bench_period_realization()["synthetic_period_realization"] == 1.0


def test_beilinson_regulator():
    assert bench_beilinson_regulator()["synthetic_beilinson_regulator"] == 1.0


def test_hodge_motive():
    assert bench_hodge_motive()["synthetic_hodge_motive"] == 1.0


def test_f_motive():
    assert bench_f_motive()["synthetic_f_motive"] == 1.0
