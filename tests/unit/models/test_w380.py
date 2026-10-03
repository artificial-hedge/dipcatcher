from quant_fund.models.baire_space import bench_baire_space
from quant_fund.models.borel_functions import bench_borel_functions
from quant_fund.models.determinacy_toy import bench_determinacy_toy
from quant_fund.models.perfect_set_prop import bench_perfect_set_prop
from quant_fund.models.polish_topology import bench_polish_topology
from quant_fund.models.souslin_op import bench_souslin_op


def test_baire_space():
    assert bench_baire_space()["synthetic_baire_space"] == 1.0


def test_polish_topology():
    assert bench_polish_topology()["synthetic_polish_topology"] == 1.0


def test_borel_functions():
    assert bench_borel_functions()["synthetic_borel_functions"] == 1.0


def test_souslin_op():
    assert bench_souslin_op()["synthetic_souslin_op"] == 1.0


def test_determinacy_toy():
    assert bench_determinacy_toy()["synthetic_determinacy_toy"] == 1.0


def test_perfect_set_prop():
    assert bench_perfect_set_prop()["synthetic_perfect_set_prop"] == 1.0
