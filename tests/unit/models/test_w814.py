from quant_fund.models.karal_flow import (
    bench_karal_flow,
)
from quant_fund.models.kunita_flow import (
    bench_kunita_flow,
)
from quant_fund.models.liouville_flow import (
    bench_liouville_flow,
)
from quant_fund.models.meyers_process import (
    bench_meyers_process,
)
from quant_fund.models.stochastic_damping import (
    bench_stochastic_damping,
)
from quant_fund.models.stochastic_flow import (
    bench_stochastic_flow,
)


def test_stochastic_flow():
    assert bench_stochastic_flow()["synthetic_stochastic_flow"] == 1.0


def test_kunita_flow():
    assert bench_kunita_flow()["synthetic_kunita_flow"] == 1.0


def test_liouville_flow():
    assert bench_liouville_flow()["synthetic_liouville_flow"] == 1.0


def test_stochastic_damping():
    assert bench_stochastic_damping()["synthetic_stochastic_damping"] == 1.0


def test_meyers_process():
    assert bench_meyers_process()["synthetic_meyers_process"] == 1.0


def test_karal_flow():
    assert bench_karal_flow()["synthetic_karal_flow"] == 1.0
