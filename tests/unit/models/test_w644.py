from quant_fund.models.allday_k import bench_allday_k
from quant_fund.models.hall_alg import bench_hall_alg
from quant_fund.models.residue_k import bench_residue_k
from quant_fund.models.s_multicat import bench_s_multicat
from quant_fund.models.suslin_wagoner import (
    bench_suslin_wagoner,
)
from quant_fund.models.weibel_nil import bench_weibel_nil


def test_s_multicat():
    assert bench_s_multicat()["synthetic_s_multicat"] == 1.0


def test_allday_k():
    assert bench_allday_k()["synthetic_allday_k"] == 1.0


def test_residue_k():
    assert bench_residue_k()["synthetic_residue_k"] == 1.0


def test_suslin_wagoner():
    assert bench_suslin_wagoner()["synthetic_suslin_wagoner"] == 1.0


def test_weibel_nil():
    assert bench_weibel_nil()["synthetic_weibel_nil"] == 1.0


def test_hall_alg():
    assert bench_hall_alg()["synthetic_hall_alg"] == 1.0
