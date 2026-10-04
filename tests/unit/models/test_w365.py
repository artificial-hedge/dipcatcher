from quant_fund.models.azuma import bench_azuma
from quant_fund.models.coupling_arg import bench_coupling_arg
from quant_fund.models.doob_decomp import bench_doob_decomp
from quant_fund.models.ergodic_thm import bench_ergodic_thm
from quant_fund.models.martingale_clt import bench_martingale_clt
from quant_fund.models.optional_stopping import bench_optional_stopping


def test_optional_stopping():
    assert bench_optional_stopping()["synthetic_optional_stopping"] == 1.0


def test_doob_decomp():
    assert bench_doob_decomp()["synthetic_doob_decomp"] == 1.0


def test_martingale_clt():
    assert bench_martingale_clt()["synthetic_martingale_clt"] == 1.0


def test_azuma():
    assert bench_azuma()["synthetic_azuma"] == 1.0


def test_coupling_arg():
    assert bench_coupling_arg()["synthetic_coupling_arg"] == 1.0


def test_ergodic_thm():
    assert bench_ergodic_thm()["synthetic_ergodic_thm"] == 1.0
