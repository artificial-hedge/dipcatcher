from quant_fund.models.alexandrov_fenchel import (
    bench_alexandrov_fenchel,
)
from quant_fund.models.brunn_minkowski import (
    bench_brunn_minkowski,
)
from quant_fund.models.helly_theorem import (
    bench_helly_theorem,
)
from quant_fund.models.isoperimetric_ineq import (
    bench_isoperimetric_ineq,
)
from quant_fund.models.minkowski_sum import (
    bench_minkowski_sum,
)
from quant_fund.models.mixed_volume import (
    bench_mixed_volume,
)


def test_brunn_minkowski():
    assert bench_brunn_minkowski()["synthetic_brunn_minkowski"] == 1.0


def test_alexandrov_fenchel():
    assert bench_alexandrov_fenchel()["synthetic_alexandrov_fenchel"] == 1.0


def test_isoperimetric_ineq():
    assert bench_isoperimetric_ineq()["synthetic_isoperimetric_ineq"] == 1.0


def test_minkowski_sum():
    assert bench_minkowski_sum()["synthetic_minkowski_sum"] == 1.0


def test_mixed_volume():
    assert bench_mixed_volume()["synthetic_mixed_volume"] == 1.0


def test_helly_theorem():
    assert bench_helly_theorem()["synthetic_helly_theorem"] == 1.0
