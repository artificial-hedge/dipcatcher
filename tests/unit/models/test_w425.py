from quant_fund.models.coarse_space import bench_coarse_space
from quant_fund.models.gerbe_toy import bench_gerbe_toy
from quant_fund.models.moduli_stack import bench_moduli_stack
from quant_fund.models.quotient_stack import bench_quotient_stack
from quant_fund.models.stack_morph import bench_stack_morph
from quant_fund.models.stacky_curve import bench_stacky_curve


def test_moduli_stack():
    assert bench_moduli_stack()["synthetic_moduli_stack"] == 1.0


def test_stacky_curve():
    assert bench_stacky_curve()["synthetic_stacky_curve"] == 1.0


def test_coarse_space():
    assert bench_coarse_space()["synthetic_coarse_space"] == 1.0


def test_quotient_stack():
    assert bench_quotient_stack()["synthetic_quotient_stack"] == 1.0


def test_gerbe_toy():
    assert bench_gerbe_toy()["synthetic_gerbe_toy"] == 1.0


def test_stack_morph():
    assert bench_stack_morph()["synthetic_stack_morph"] == 1.0
