from quant_fund.models.acl_closure import bench_acl_closure
from quant_fund.models.morley_rank import bench_morley_rank
from quant_fund.models.omega_categoricity import bench_omega_categoricity
from quant_fund.models.quantifier_elim import bench_quantifier_elim
from quant_fund.models.realize_types import bench_realize_types
from quant_fund.models.vocab_interp import bench_vocab_interp


def test_quantifier_elim():
    assert bench_quantifier_elim()["synthetic_quantifier_elim"] == 1.0


def test_realize_types():
    assert bench_realize_types()["synthetic_realize_types"] == 1.0


def test_omega_categoricity():
    assert bench_omega_categoricity()["synthetic_omega_categoricity"] == 1.0


def test_acl_closure():
    assert bench_acl_closure()["synthetic_acl_closure"] == 1.0


def test_morley_rank():
    assert bench_morley_rank()["synthetic_morley_rank"] == 1.0


def test_vocab_interp():
    assert bench_vocab_interp()["synthetic_vocab_interp"] == 1.0
