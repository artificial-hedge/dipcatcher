from quant_fund.models.homotopy_abelian import (
    bench_homotopy_abelian,
)
from quant_fund.models.homotopy_extended import (
    bench_homotopy_extended,
)
from quant_fund.models.homotopy_finite import (
    bench_homotopy_finite,
)
from quant_fund.models.homotopy_infinite import (
    bench_homotopy_infinite,
)
from quant_fund.models.stable_compact import (
    bench_stable_compact,
)
from quant_fund.models.stable_synthetic import (
    bench_stable_synthetic,
)


def test_homotopy_abelian():
    assert bench_homotopy_abelian()["synthetic_homotopy_abelian"] == 1.0


def test_homotopy_finite():
    assert bench_homotopy_finite()["synthetic_homotopy_finite"] == 1.0


def test_homotopy_infinite():
    assert bench_homotopy_infinite()["synthetic_homotopy_infinite"] == 1.0


def test_homotopy_extended():
    assert bench_homotopy_extended()["synthetic_homotopy_extended"] == 1.0


def test_stable_synthetic():
    assert bench_stable_synthetic()["synthetic_stable_synthetic"] == 1.0


def test_stable_compact():
    assert bench_stable_compact()["synthetic_stable_compact"] == 1.0
