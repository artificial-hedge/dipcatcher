from quant_fund.models.chow_motive import bench_chow_motive
from quant_fund.models.nori_motive import bench_nori_motive
from quant_fund.models.num_equiv import bench_num_equiv
from quant_fund.models.standard_conj import bench_standard_conj
from quant_fund.models.tate_motive import bench_tate_motive
from quant_fund.models.voev_motive import bench_voev_motive


def test_chow_motive():
    assert bench_chow_motive()["synthetic_chow_motive"] == 1.0


def test_nori_motive():
    assert bench_nori_motive()["synthetic_nori_motive"] == 1.0


def test_num_equiv():
    assert bench_num_equiv()["synthetic_num_equiv"] == 1.0


def test_standard_conj():
    assert bench_standard_conj()["synthetic_standard_conj"] == 1.0


def test_voev_motive():
    assert bench_voev_motive()["synthetic_voev_motive"] == 1.0


def test_tate_motive():
    assert bench_tate_motive()["synthetic_tate_motive"] == 1.0
