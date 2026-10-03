from quant_fund.models.condensed_coh import (
    bench_condensed_coh,
)
from quant_fund.models.condensed_ring import (
    bench_condensed_ring,
)
from quant_fund.models.discrete_liquid import (
    bench_discrete_liquid,
)
from quant_fund.models.liquid_ring import (
    bench_liquid_ring,
)
from quant_fund.models.scholze_trace import (
    bench_scholze_trace,
)
from quant_fund.models.smith_project import (
    bench_smith_project,
)


def test_discrete_liquid():
    assert bench_discrete_liquid()["synthetic_discrete_liquid"] == 1.0


def test_smith_project():
    assert bench_smith_project()["synthetic_smith_project"] == 1.0


def test_condensed_ring():
    assert bench_condensed_ring()["synthetic_condensed_ring"] == 1.0


def test_liquid_ring():
    assert bench_liquid_ring()["synthetic_liquid_ring"] == 1.0


def test_scholze_trace():
    assert bench_scholze_trace()["synthetic_scholze_trace"] == 1.0


def test_condensed_coh():
    assert bench_condensed_coh()["synthetic_condensed_coh"] == 1.0
