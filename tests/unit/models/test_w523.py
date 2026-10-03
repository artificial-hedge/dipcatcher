from quant_fund.models.erdos_distinct import bench_erdos_distinct
from quant_fund.models.ff_kakeya import bench_ff_kakeya
from quant_fund.models.guth_katz import bench_guth_katz
from quant_fund.models.joints_thm import bench_joints_thm
from quant_fund.models.kakeya import bench_kakeya
from quant_fund.models.sz_trotter import bench_sz_trotter


def test_erdos_distinct():
    assert bench_erdos_distinct()["synthetic_erdos_distinct"] == 1.0


def test_sz_trotter():
    assert bench_sz_trotter()["synthetic_sz_trotter"] == 1.0


def test_kakeya():
    assert bench_kakeya()["synthetic_kakeya"] == 1.0


def test_ff_kakeya():
    assert bench_ff_kakeya()["synthetic_ff_kakeya"] == 1.0


def test_joints_thm():
    assert bench_joints_thm()["synthetic_joints_thm"] == 1.0


def test_guth_katz():
    assert bench_guth_katz()["synthetic_guth_katz"] == 1.0
