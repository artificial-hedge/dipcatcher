from quant_fund.models.balanced_dd import (
    bench_balanced_dd,
)
from quant_fund.models.diagonal_scale import (
    bench_diagonal_scale,
)
from quant_fund.models.nonoverlap_dd import (
    bench_nonoverlap_dd,
)
from quant_fund.models.overlap_dd import (
    bench_overlap_dd,
)
from quant_fund.models.restrictive_dd import (
    bench_restrictive_dd,
)
from quant_fund.models.spai_precond import (
    bench_spai_precond,
)


def test_spai_precond():
    assert bench_spai_precond()["synthetic_spai_precond"] == 1.0


def test_diagonal_scale():
    assert bench_diagonal_scale()["synthetic_diagonal_scale"] == 1.0


def test_nonoverlap_dd():
    assert bench_nonoverlap_dd()["synthetic_nonoverlap_dd"] == 1.0


def test_overlap_dd():
    assert bench_overlap_dd()["synthetic_overlap_dd"] == 1.0


def test_restrictive_dd():
    assert bench_restrictive_dd()["synthetic_restrictive_dd"] == 1.0


def test_balanced_dd():
    assert bench_balanced_dd()["synthetic_balanced_dd"] == 1.0
