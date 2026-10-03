from quant_fund.models.descriptive3 import bench_descriptive3
from quant_fund.models.forcing2 import bench_forcing2
from quant_fund.models.inner_model import bench_inner_model
from quant_fund.models.ordinal_notation import bench_ordinal_notation
from quant_fund.models.proof_mining import bench_proof_mining
from quant_fund.models.recursion3 import bench_recursion3


def test_forcing2():
    assert bench_forcing2()["synthetic_forcing2"] == 1.0


def test_inner_model():
    assert bench_inner_model()["synthetic_inner_model"] == 1.0


def test_descriptive3():
    assert bench_descriptive3()["synthetic_descriptive3"] == 1.0


def test_recursion3():
    assert bench_recursion3()["synthetic_recursion3"] == 1.0


def test_proof_mining():
    assert bench_proof_mining()["synthetic_proof_mining"] == 1.0


def test_ordinal_notation():
    assert bench_ordinal_notation()["synthetic_ordinal_notation"] == 1.0
