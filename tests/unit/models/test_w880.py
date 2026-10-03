from quant_fund.models.adaptive_quad2 import (
    bench_adaptive_quad2,
)
from quant_fund.models.cubature_rule import (
    bench_cubature_rule,
)
from quant_fund.models.empirical_interp import (
    bench_empirical_interp,
)
from quant_fund.models.gq_adaptive import (
    bench_gq_adaptive,
)
from quant_fund.models.pod_deim import (
    bench_pod_deim,
)
from quant_fund.models.tensor_interp import (
    bench_tensor_interp,
)


def test_gq_adaptive():
    assert bench_gq_adaptive()["synthetic_gq_adaptive"] == 1.0


def test_adaptive_quad2():
    assert bench_adaptive_quad2()["synthetic_adaptive_quad2"] == 1.0


def test_pod_deim():
    assert bench_pod_deim()["synthetic_pod_deim"] == 1.0


def test_empirical_interp():
    assert bench_empirical_interp()["synthetic_empirical_interp"] == 1.0


def test_cubature_rule():
    assert bench_cubature_rule()["synthetic_cubature_rule"] == 1.0


def test_tensor_interp():
    assert bench_tensor_interp()["synthetic_tensor_interp"] == 1.0
