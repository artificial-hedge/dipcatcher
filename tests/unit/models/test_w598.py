from quant_fund.models.bloch_beilinson import (
    bench_bloch_beilinson,
)
from quant_fund.models.borel_regulator import (
    bench_borel_regulator,
)
from quant_fund.models.etale_ktheory import (
    bench_etale_ktheory,
)
from quant_fund.models.lichtenbaum_k import (
    bench_lichtenbaum_k,
)
from quant_fund.models.soul_elem import bench_soul_elem
from quant_fund.models.thh_trace import bench_thh_trace


def test_borel_regulator():
    assert bench_borel_regulator()["synthetic_borel_regulator"] == 1.0


def test_soul_elem():
    assert bench_soul_elem()["synthetic_soul_elem"] == 1.0


def test_lichtenbaum_k():
    assert bench_lichtenbaum_k()["synthetic_lichtenbaum_k"] == 1.0


def test_bloch_beilinson():
    assert bench_bloch_beilinson()["synthetic_bloch_beilinson"] == 1.0


def test_etale_ktheory():
    assert bench_etale_ktheory()["synthetic_etale_ktheory"] == 1.0


def test_thh_trace():
    assert bench_thh_trace()["synthetic_thh_trace"] == 1.0
