from quant_fund.models.dedekind_zeta import bench_dedekind_zeta
from quant_fund.models.dirichlet_unit import bench_dirichlet_unit
from quant_fund.models.ideal_class import bench_ideal_class
from quant_fund.models.minkowski_bound import bench_minkowski_bound
from quant_fund.models.regulator import bench_regulator
from quant_fund.models.splitting_prime import bench_splitting_prime


def test_dirichlet_unit():
    assert bench_dirichlet_unit()["synthetic_dirichlet_unit"] == 1.0


def test_regulator():
    assert bench_regulator()["synthetic_regulator"] == 1.0


def test_ideal_class():
    assert bench_ideal_class()["synthetic_ideal_class"] == 1.0


def test_minkowski_bound():
    assert bench_minkowski_bound()["synthetic_minkowski_bound"] == 1.0


def test_dedekind_zeta():
    assert bench_dedekind_zeta()["synthetic_dedekind_zeta"] == 1.0


def test_splitting_prime():
    assert bench_splitting_prime()["synthetic_splitting_prime"] == 1.0
