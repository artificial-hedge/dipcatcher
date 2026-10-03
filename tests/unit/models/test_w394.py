from quant_fund.models.birkhoff_rep import bench_birkhoff_rep
from quant_fund.models.dilworth_partition import bench_dilworth_partition
from quant_fund.models.downset_lattice import bench_downset_lattice
from quant_fund.models.linear_extension import bench_linear_extension
from quant_fund.models.sperner_bound import bench_sperner_bound
from quant_fund.models.zeta_mobius import bench_zeta_mobius


def test_downset_lattice():
    assert bench_downset_lattice()["synthetic_downset_lattice"] == 1.0


def test_zeta_mobius():
    assert bench_zeta_mobius()["synthetic_zeta_mobius"] == 1.0


def test_linear_extension():
    assert bench_linear_extension()["synthetic_linear_extension"] == 1.0


def test_sperner_bound():
    assert bench_sperner_bound()["synthetic_sperner_bound"] == 1.0


def test_dilworth_partition():
    assert bench_dilworth_partition()["synthetic_dilworth_partition"] == 1.0


def test_birkhoff_rep():
    assert bench_birkhoff_rep()["synthetic_birkhoff_rep"] == 1.0
