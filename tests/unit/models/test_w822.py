from quant_fund.models.enlargement_f import (
    bench_enlargement_f,
)
from quant_fund.models.initial_enlarg import (
    bench_initial_enlarg,
)
from quant_fund.models.natural_filtration import (
    bench_natural_filtration,
)
from quant_fund.models.progressive_enlarg import (
    bench_progressive_enlarg,
)
from quant_fund.models.right_continuous_f import (
    bench_right_continuous_f,
)
from quant_fund.models.usual_aug import (
    bench_usual_aug,
)


def test_natural_filtration():
    assert bench_natural_filtration()["synthetic_natural_filtration"] == 1.0


def test_right_continuous_f():
    assert bench_right_continuous_f()["synthetic_right_continuous_f"] == 1.0


def test_usual_aug():
    assert bench_usual_aug()["synthetic_usual_aug"] == 1.0


def test_enlargement_f():
    assert bench_enlargement_f()["synthetic_enlargement_f"] == 1.0


def test_initial_enlarg():
    assert bench_initial_enlarg()["synthetic_initial_enlarg"] == 1.0


def test_progressive_enlarg():
    assert bench_progressive_enlarg()["synthetic_progressive_enlarg"] == 1.0
