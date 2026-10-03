from quant_fund.models.dolean_mart import (
    bench_dolean_mart,
)
from quant_fund.models.follmer_mart import (
    bench_follmer_mart,
)
from quant_fund.models.local_mart2 import (
    bench_local_mart2,
)
from quant_fund.models.protter_ito import (
    bench_protter_ito,
)
from quant_fund.models.strong_sol import bench_strong_sol
from quant_fund.models.usual_cond import bench_usual_cond


def test_usual_cond():
    assert bench_usual_cond()["synthetic_usual_cond"] == 1.0


def test_dolean_mart():
    assert bench_dolean_mart()["synthetic_dolean_mart"] == 1.0


def test_strong_sol():
    assert bench_strong_sol()["synthetic_strong_sol"] == 1.0


def test_local_mart2():
    assert bench_local_mart2()["synthetic_local_mart2"] == 1.0


def test_follmer_mart():
    assert bench_follmer_mart()["synthetic_follmer_mart"] == 1.0


def test_protter_ito():
    assert bench_protter_ito()["synthetic_protter_ito"] == 1.0
