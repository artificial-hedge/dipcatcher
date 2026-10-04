from quant_fund.models.dominated_split import bench_dominated_split
from quant_fund.models.katok_horseshoe import bench_katok_horseshoe
from quant_fund.models.lyapunov_chart import bench_lyapunov_chart
from quant_fund.models.nonuniform_hyp import bench_nonuniform_hyp
from quant_fund.models.osceledets_reg import bench_osceledets_reg
from quant_fund.models.pesin_theory import bench_pesin_theory


def test_pesin_theory():
    assert bench_pesin_theory()["synthetic_pesin_theory"] == 1.0


def test_nonuniform_hyp():
    assert bench_nonuniform_hyp()["synthetic_nonuniform_hyp"] == 1.0


def test_dominated_split():
    assert bench_dominated_split()["synthetic_dominated_split"] == 1.0


def test_osceledets_reg():
    assert bench_osceledets_reg()["synthetic_osceledets_reg"] == 1.0


def test_lyapunov_chart():
    assert bench_lyapunov_chart()["synthetic_lyapunov_chart"] == 1.0


def test_katok_horseshoe():
    assert bench_katok_horseshoe()["synthetic_katok_horseshoe"] == 1.0
