from quant_fund.models.bounded_mart import (
    bench_bounded_mart,
)
from quant_fund.models.cadlag_mart import (
    bench_cadlag_mart,
)
from quant_fund.models.decomp_mart import (
    bench_decomp_mart,
)
from quant_fund.models.fv_mart import (
    bench_fv_mart,
)
from quant_fund.models.local_time_process import (
    bench_local_time_process,
)
from quant_fund.models.locator_proc import (
    bench_locator_proc,
)


def test_local_time_process():
    assert bench_local_time_process()["synthetic_local_time_process"] == 1.0


def test_bounded_mart():
    assert bench_bounded_mart()["synthetic_bounded_mart"] == 1.0


def test_fv_mart():
    assert bench_fv_mart()["synthetic_fv_mart"] == 1.0


def test_cadlag_mart():
    assert bench_cadlag_mart()["synthetic_cadlag_mart"] == 1.0


def test_locator_proc():
    assert bench_locator_proc()["synthetic_locator_proc"] == 1.0


def test_decomp_mart():
    assert bench_decomp_mart()["synthetic_decomp_mart"] == 1.0
