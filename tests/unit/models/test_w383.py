from quant_fund.models.co_homology import bench_co_homology
from quant_fund.models.em_space import bench_em_space
from quant_fund.models.loop_space import bench_loop_space
from quant_fund.models.mapping_cone import bench_mapping_cone
from quant_fund.models.stiefel_whitney import bench_stiefel_whitney
from quant_fund.models.transfer import bench_transfer


def test_mapping_cone():
    assert bench_mapping_cone()["synthetic_mapping_cone"] == 1.0


def test_loop_space():
    assert bench_loop_space()["synthetic_loop_space"] == 1.0


def test_em_space():
    assert bench_em_space()["synthetic_em_space"] == 1.0


def test_co_homology():
    assert bench_co_homology()["synthetic_co_homology"] == 1.0


def test_stiefel_whitney():
    assert bench_stiefel_whitney()["synthetic_stiefel_whitney"] == 1.0


def test_transfer():
    assert bench_transfer()["synthetic_transfer"] == 1.0
