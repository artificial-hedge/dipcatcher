from quant_fund.models.antithetic_var import (
    bench_antithetic_var,
)
from quant_fund.models.common_random import (
    bench_common_random,
)
from quant_fund.models.conditional_mc import (
    bench_conditional_mc,
)
from quant_fund.models.control_variate import (
    bench_control_variate,
)
from quant_fund.models.importance_sampling import (
    bench_importance_sampling,
)
from quant_fund.models.stratified_var import (
    bench_stratified_var,
)


def test_antithetic_var():
    assert bench_antithetic_var()["synthetic_antithetic_var"] == 1.0


def test_control_variate():
    assert bench_control_variate()["synthetic_control_variate"] == 1.0


def test_importance_sampling():
    assert bench_importance_sampling()["synthetic_importance_sampling"] == 1.0


def test_stratified_var():
    assert bench_stratified_var()["synthetic_stratified_var"] == 1.0


def test_common_random():
    assert bench_common_random()["synthetic_common_random"] == 1.0


def test_conditional_mc():
    assert bench_conditional_mc()["synthetic_conditional_mc"] == 1.0
