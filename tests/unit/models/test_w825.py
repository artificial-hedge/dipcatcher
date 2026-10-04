from quant_fund.models.castaing_rep import (
    bench_castaing_rep,
)
from quant_fund.models.integrand_map import (
    bench_integrand_map,
)
from quant_fund.models.kura_ryll import (
    bench_kura_ryll,
)
from quant_fund.models.measur_select import (
    bench_measur_select,
)
from quant_fund.models.measurable_graph import (
    bench_measurable_graph,
)
from quant_fund.models.stoch_open import (
    bench_stoch_open,
)


def test_measur_select():
    assert bench_measur_select()["synthetic_measur_select"] == 1.0


def test_kura_ryll():
    assert bench_kura_ryll()["synthetic_kura_ryll"] == 1.0


def test_castaing_rep():
    assert bench_castaing_rep()["synthetic_castaing_rep"] == 1.0


def test_measurable_graph():
    assert bench_measurable_graph()["synthetic_measurable_graph"] == 1.0


def test_integrand_map():
    assert bench_integrand_map()["synthetic_integrand_map"] == 1.0


def test_stoch_open():
    assert bench_stoch_open()["synthetic_stoch_open"] == 1.0
