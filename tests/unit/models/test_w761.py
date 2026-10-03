from quant_fund.models.alternating_renewal import (
    bench_alternating_renewal,
)
from quant_fund.models.blackwell_renewal import (
    bench_blackwell_renewal,
)
from quant_fund.models.delayed_renewal import bench_delayed_renewal
from quant_fund.models.excess_renewal import bench_excess_renewal
from quant_fund.models.key_renewal import bench_key_renewal
from quant_fund.models.renewal_reward2 import (
    bench_renewal_reward2,
)


def test_blackwell_renewal():
    assert bench_blackwell_renewal()["synthetic_blackwell_renewal"] == 1.0


def test_key_renewal():
    assert bench_key_renewal()["synthetic_key_renewal"] == 1.0


def test_excess_renewal():
    assert bench_excess_renewal()["synthetic_excess_renewal"] == 1.0


def test_alternating_renewal():
    assert bench_alternating_renewal()["synthetic_alternating_renewal"] == 1.0


def test_renewal_reward2():
    assert bench_renewal_reward2()["synthetic_renewal_reward2"] == 1.0


def test_delayed_renewal():
    assert bench_delayed_renewal()["synthetic_delayed_renewal"] == 1.0
