from quant_fund.models.bounded_var import (
    bench_bounded_var,
)
from quant_fund.models.covariation import (
    bench_covariation,
)
from quant_fund.models.ito_integral import (
    bench_ito_integral,
)
from quant_fund.models.mart_meas import (
    bench_mart_meas,
)
from quant_fund.models.stochastic_int2 import (
    bench_stochastic_int2,
)
from quant_fund.models.vector_mart import (
    bench_vector_mart,
)


def test_ito_integral():
    assert bench_ito_integral()["synthetic_ito_integral"] == 1.0


def test_mart_meas():
    assert bench_mart_meas()["synthetic_mart_meas"] == 1.0


def test_vector_mart():
    assert bench_vector_mart()["synthetic_vector_mart"] == 1.0


def test_bounded_var():
    assert bench_bounded_var()["synthetic_bounded_var"] == 1.0


def test_stochastic_int2():
    assert bench_stochastic_int2()["synthetic_stochastic_int2"] == 1.0


def test_covariation():
    assert bench_covariation()["synthetic_covariation"] == 1.0
