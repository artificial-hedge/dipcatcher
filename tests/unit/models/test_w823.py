from quant_fund.models.cylindrical_law import (
    bench_cylindrical_law,
)
from quant_fund.models.finite_dim import (
    bench_finite_dim,
)
from quant_fund.models.law_convergence import (
    bench_law_convergence,
)
from quant_fund.models.polish_law import (
    bench_polish_law,
)
from quant_fund.models.support_law import (
    bench_support_law,
)
from quant_fund.models.tight_law import (
    bench_tight_law,
)


def test_support_law():
    assert bench_support_law()["synthetic_support_law"] == 1.0


def test_polish_law():
    assert bench_polish_law()["synthetic_polish_law"] == 1.0


def test_tight_law():
    assert bench_tight_law()["synthetic_tight_law"] == 1.0


def test_law_convergence():
    assert bench_law_convergence()["synthetic_law_convergence"] == 1.0


def test_finite_dim():
    assert bench_finite_dim()["synthetic_finite_dim"] == 1.0


def test_cylindrical_law():
    assert bench_cylindrical_law()["synthetic_cylindrical_law"] == 1.0
