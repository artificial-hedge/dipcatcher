from quant_fund.models.bulk_queue import bench_bulk_queue
from quant_fund.models.gm_queue import bench_gm_queue
from quant_fund.models.mg1_queue import bench_mg1_queue
from quant_fund.models.mm1_queue import bench_mm1_queue
from quant_fund.models.priority_queue import bench_priority_queue
from quant_fund.models.retrial_queue import bench_retrial_queue


def test_mm1_queue():
    assert bench_mm1_queue()["synthetic_mm1_queue"] == 1.0


def test_mg1_queue():
    assert bench_mg1_queue()["synthetic_mg1_queue"] == 1.0


def test_gm_queue():
    assert bench_gm_queue()["synthetic_gm_queue"] == 1.0


def test_bulk_queue():
    assert bench_bulk_queue()["synthetic_bulk_queue"] == 1.0


def test_retrial_queue():
    assert bench_retrial_queue()["synthetic_retrial_queue"] == 1.0


def test_priority_queue():
    assert bench_priority_queue()["synthetic_priority_queue"] == 1.0
