from quant_fund.models.homotopy_general import (
    bench_homotopy_general,
)
from quant_fund.models.homotopy_rational import (
    bench_homotopy_rational,
)
from quant_fund.models.stable_dual import bench_stable_dual
from quant_fund.models.stable_lie import bench_stable_lie
from quant_fund.models.stable_motivic import (
    bench_stable_motivic,
)
from quant_fund.models.stable_perf import bench_stable_perf


def test_homotopy_general():
    assert bench_homotopy_general()["synthetic_homotopy_general"] == 1.0


def test_homotopy_rational():
    assert bench_homotopy_rational()["synthetic_homotopy_rational"] == 1.0


def test_stable_dual():
    assert bench_stable_dual()["synthetic_stable_dual"] == 1.0


def test_stable_lie():
    assert bench_stable_lie()["synthetic_stable_lie"] == 1.0


def test_stable_motivic():
    assert bench_stable_motivic()["synthetic_stable_motivic"] == 1.0


def test_stable_perf():
    assert bench_stable_perf()["synthetic_stable_perf"] == 1.0
