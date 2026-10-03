from quant_fund.models.covering_number import (
    bench_covering_number,
)
from quant_fund.models.entropy_integral import (
    bench_entropy_integral,
)
from quant_fund.models.metric_entropy import (
    bench_metric_entropy,
)
from quant_fund.models.rademacher_cplx import (
    bench_rademacher_cplx,
)
from quant_fund.models.symmetrization import (
    bench_symmetrization,
)
from quant_fund.models.uniform_clt import (
    bench_uniform_clt,
)


def test_entropy_integral():
    assert bench_entropy_integral()["synthetic_entropy_integral"] == 1.0


def test_uniform_clt():
    assert bench_uniform_clt()["synthetic_uniform_clt"] == 1.0


def test_symmetrization():
    assert bench_symmetrization()["synthetic_symmetrization"] == 1.0


def test_rademacher_cplx():
    assert bench_rademacher_cplx()["synthetic_rademacher_cplx"] == 1.0


def test_covering_number():
    assert bench_covering_number()["synthetic_covering_number"] == 1.0


def test_metric_entropy():
    assert bench_metric_entropy()["synthetic_metric_entropy"] == 1.0
