from quant_fund.models.fqmotive import bench_fqmotive
from quant_fund.models.higher_chow2 import bench_higher_chow2
from quant_fund.models.motivic_chern import bench_motivic_chern
from quant_fund.models.motivic_landin import bench_motivic_landin
from quant_fund.models.mtc_motive import bench_mtc_motive
from quant_fund.models.triang_motive import bench_triang_motive


def test_mtc_motive():
    assert bench_mtc_motive()["synthetic_mtc_motive"] == 1.0


def test_fqmotive():
    assert bench_fqmotive()["synthetic_fqmotive"] == 1.0


def test_triang_motive():
    assert bench_triang_motive()["synthetic_triang_motive"] == 1.0


def test_motivic_chern():
    assert bench_motivic_chern()["synthetic_motivic_chern"] == 1.0


def test_motivic_landin():
    assert bench_motivic_landin()["synthetic_motivic_landin"] == 1.0


def test_higher_chow2():
    assert bench_higher_chow2()["synthetic_higher_chow2"] == 1.0
