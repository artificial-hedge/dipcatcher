from quant_fund.models.motivic_crystal import bench_motivic_crystal
from quant_fund.models.motivic_cycle import bench_motivic_cycle
from quant_fund.models.motivic_etale import bench_motivic_etale
from quant_fund.models.motivic_prism import bench_motivic_prism
from quant_fund.models.motivic_sphere3 import bench_motivic_sphere3
from quant_fund.models.motivic_tower import bench_motivic_tower


def test_motivic_tower():
    assert bench_motivic_tower()["synthetic_motivic_tower"] == 1.0


def test_motivic_sphere3():
    assert bench_motivic_sphere3()["synthetic_motivic_sphere3"] == 1.0


def test_motivic_etale():
    assert bench_motivic_etale()["synthetic_motivic_etale"] == 1.0


def test_motivic_crystal():
    assert bench_motivic_crystal()["synthetic_motivic_crystal"] == 1.0


def test_motivic_prism():
    assert bench_motivic_prism()["synthetic_motivic_prism"] == 1.0


def test_motivic_cycle():
    assert bench_motivic_cycle()["synthetic_motivic_cycle"] == 1.0
