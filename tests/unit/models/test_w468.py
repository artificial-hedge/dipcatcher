from quant_fund.models.abstract_erc import bench_abstract_erc
from quant_fund.models.nip_theory import bench_nip_theory
from quant_fund.models.nonforking import bench_nonforking
from quant_fund.models.o_minimal import bench_o_minimal
from quant_fund.models.simple_theory import bench_simple_theory
from quant_fund.models.tame_metric import bench_tame_metric


def test_o_minimal():
    assert bench_o_minimal()["synthetic_o_minimal"] == 1.0


def test_nip_theory():
    assert bench_nip_theory()["synthetic_nip_theory"] == 1.0


def test_nonforking():
    assert bench_nonforking()["synthetic_nonforking"] == 1.0


def test_simple_theory():
    assert bench_simple_theory()["synthetic_simple_theory"] == 1.0


def test_abstract_erc():
    assert bench_abstract_erc()["synthetic_abstract_erc"] == 1.0


def test_tame_metric():
    assert bench_tame_metric()["synthetic_tame_metric"] == 1.0
