from quant_fund.models.first_order_rel import (
    bench_first_order_rel,
)
from quant_fund.models.line_sampling import (
    bench_line_sampling,
)
from quant_fund.models.metamodel_rel import (
    bench_metamodel_rel,
)
from quant_fund.models.sorm_method import (
    bench_sorm_method,
)
from quant_fund.models.subset_sim import (
    bench_subset_sim,
)
from quant_fund.models.uq_reliability import (
    bench_uq_reliability,
)


def test_uq_reliability():
    assert bench_uq_reliability()["synthetic_uq_reliability"] == 1.0


def test_first_order_rel():
    assert bench_first_order_rel()["synthetic_first_order_rel"] == 1.0


def test_sorm_method():
    assert bench_sorm_method()["synthetic_sorm_method"] == 1.0


def test_subset_sim():
    assert bench_subset_sim()["synthetic_subset_sim"] == 1.0


def test_line_sampling():
    assert bench_line_sampling()["synthetic_line_sampling"] == 1.0


def test_metamodel_rel():
    assert bench_metamodel_rel()["synthetic_metamodel_rel"] == 1.0
