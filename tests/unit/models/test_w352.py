from quant_fund.models.cartan_matrix import bench_cartan_matrix
from quant_fund.models.killing_form import bench_killing_form
from quant_fund.models.root_lattice_a2 import bench_root_lattice_a2
from quant_fund.models.sl2_structure import bench_sl2_structure
from quant_fund.models.su2_algebra import bench_su2_algebra
from quant_fund.models.weyl_group_a2 import bench_weyl_group_a2


def test_cartan_matrix():
    assert bench_cartan_matrix()["synthetic_cartan_matrix"] == 1.0


def test_weyl_group_a2():
    assert bench_weyl_group_a2()["synthetic_weyl_group_a2"] == 1.0


def test_killing_form():
    assert bench_killing_form()["synthetic_killing_form"] == 1.0


def test_root_lattice_a2():
    assert bench_root_lattice_a2()["synthetic_root_lattice_a2"] == 1.0


def test_sl2_structure():
    assert bench_sl2_structure()["synthetic_sl2_structure"] == 1.0


def test_su2_algebra():
    assert bench_su2_algebra()["synthetic_su2_algebra"] == 1.0
