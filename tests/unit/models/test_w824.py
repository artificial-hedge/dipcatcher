from quant_fund.models.convex_order import (
    bench_convex_order,
)
from quant_fund.models.first_order_dom import (
    bench_first_order_dom,
)
from quant_fund.models.hazard_rate_order import (
    bench_hazard_rate_order,
)
from quant_fund.models.second_order_dom import (
    bench_second_order_dom,
)
from quant_fund.models.supermodular_order import (
    bench_supermodular_order,
)
from quant_fund.models.usual_stoch_order import (
    bench_usual_stoch_order,
)


def test_usual_stoch_order():
    assert bench_usual_stoch_order()["synthetic_usual_stoch_order"] == 1.0


def test_first_order_dom():
    assert bench_first_order_dom()["synthetic_first_order_dom"] == 1.0


def test_second_order_dom():
    assert bench_second_order_dom()["synthetic_second_order_dom"] == 1.0


def test_convex_order():
    assert bench_convex_order()["synthetic_convex_order"] == 1.0


def test_hazard_rate_order():
    assert bench_hazard_rate_order()["synthetic_hazard_rate_order"] == 1.0


def test_supermodular_order():
    assert bench_supermodular_order()["synthetic_supermodular_order"] == 1.0
