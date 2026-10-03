from quant_fund.models.cramer_wold import (
    bench_cramer_wold,
)
from quant_fund.models.hazard_order import (
    bench_hazard_order,
)
from quant_fund.models.likelihood_order import (
    bench_likelihood_order,
)
from quant_fund.models.predictable_bracket import (
    bench_predictable_bracket,
)
from quant_fund.models.semi_mart import bench_semi_mart
from quant_fund.models.stricker_thm import (
    bench_stricker_thm,
)


def test_semi_mart():
    assert bench_semi_mart()["synthetic_semi_mart"] == 1.0


def test_predictable_bracket():
    assert bench_predictable_bracket()["synthetic_predictable_bracket"] == 1.0


def test_cramer_wold():
    assert bench_cramer_wold()["synthetic_cramer_wold"] == 1.0


def test_stricker_thm():
    assert bench_stricker_thm()["synthetic_stricker_thm"] == 1.0


def test_likelihood_order():
    assert bench_likelihood_order()["synthetic_likelihood_order"] == 1.0


def test_hazard_order():
    assert bench_hazard_order()["synthetic_hazard_order"] == 1.0
