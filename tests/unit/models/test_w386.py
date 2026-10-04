from quant_fund.models.godel_incomp import bench_godel_incomp
from quant_fund.models.interp_proof import bench_interp_proof
from quant_fund.models.modal_completeness import bench_modal_completeness
from quant_fund.models.natural_ded import bench_natural_ded
from quant_fund.models.proof_complexity import bench_proof_complexity
from quant_fund.models.sequent_calculus import bench_sequent_calculus


def test_sequent_calculus():
    assert bench_sequent_calculus()["synthetic_sequent_calculus"] == 1.0


def test_natural_ded():
    assert bench_natural_ded()["synthetic_natural_ded"] == 1.0


def test_godel_incomp():
    assert bench_godel_incomp()["synthetic_godel_incomp"] == 1.0


def test_interp_proof():
    assert bench_interp_proof()["synthetic_interp_proof"] == 1.0


def test_proof_complexity():
    assert bench_proof_complexity()["synthetic_proof_complexity"] == 1.0


def test_modal_completeness():
    assert bench_modal_completeness()["synthetic_modal_completeness"] == 1.0
