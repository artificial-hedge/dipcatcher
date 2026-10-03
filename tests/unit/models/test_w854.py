from quant_fund.models.gaussian_rbf import (
    bench_gaussian_rbf,
)
from quant_fund.models.kansa_collocation import (
    bench_kansa_collocation,
)
from quant_fund.models.multiquadric_rbf import (
    bench_multiquadric_rbf,
)
from quant_fund.models.rbf_finite_diff import (
    bench_rbf_finite_diff,
)
from quant_fund.models.rbf_interp import (
    bench_rbf_interp,
)
from quant_fund.models.wendland_rbf import (
    bench_wendland_rbf,
)


def test_rbf_interp():
    assert bench_rbf_interp()["synthetic_rbf_interp"] == 1.0


def test_gaussian_rbf():
    assert bench_gaussian_rbf()["synthetic_gaussian_rbf"] == 1.0


def test_multiquadric_rbf():
    assert bench_multiquadric_rbf()["synthetic_multiquadric_rbf"] == 1.0


def test_kansa_collocation():
    assert bench_kansa_collocation()["synthetic_kansa_collocation"] == 1.0


def test_rbf_finite_diff():
    assert bench_rbf_finite_diff()["synthetic_rbf_finite_diff"] == 1.0


def test_wendland_rbf():
    assert bench_wendland_rbf()["synthetic_wendland_rbf"] == 1.0
