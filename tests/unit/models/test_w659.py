from quant_fund.models.absolute_cohom import bench_absolute_cohom
from quant_fund.models.motivic_pairing import bench_motivic_pairing
from quant_fund.models.motivic_tate2 import bench_motivic_tate2
from quant_fund.models.motivic_weight import bench_motivic_weight
from quant_fund.models.norimotive2 import bench_norimotive2
from quant_fund.models.tate_triple import bench_tate_triple


def test_norimotive2():
    assert bench_norimotive2()["synthetic_norimotive2"] == 1.0


def test_motivic_tate2():
    assert bench_motivic_tate2()["synthetic_motivic_tate2"] == 1.0


def test_absolute_cohom():
    assert bench_absolute_cohom()["synthetic_absolute_cohom"] == 1.0


def test_motivic_weight():
    assert bench_motivic_weight()["synthetic_motivic_weight"] == 1.0


def test_tate_triple():
    assert bench_tate_triple()["synthetic_tate_triple"] == 1.0


def test_motivic_pairing():
    assert bench_motivic_pairing()["synthetic_motivic_pairing"] == 1.0
