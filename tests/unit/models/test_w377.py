from quant_fund.models.endomorphism_op import bench_endomorphism_op
from quant_fund.models.little_discs import bench_little_discs
from quant_fund.models.may_recognition import bench_may_recognition
from quant_fund.models.operad_assoc import bench_operad_assoc
from quant_fund.models.operad_comm import bench_operad_comm
from quant_fund.models.operad_tree import bench_operad_tree


def test_operad_assoc():
    assert bench_operad_assoc()["synthetic_operad_assoc"] == 1.0


def test_operad_comm():
    assert bench_operad_comm()["synthetic_operad_comm"] == 1.0


def test_little_discs():
    assert bench_little_discs()["synthetic_little_discs"] == 1.0


def test_operad_tree():
    assert bench_operad_tree()["synthetic_operad_tree"] == 1.0


def test_endomorphism_op():
    assert bench_endomorphism_op()["synthetic_endomorphism_op"] == 1.0


def test_may_recognition():
    assert bench_may_recognition()["synthetic_may_recognition"] == 1.0
