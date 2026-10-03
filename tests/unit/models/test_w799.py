from quant_fund.models.dupire_functional import (
    bench_dupire_functional,
)
from quant_fund.models.functional_ito import (
    bench_functional_ito,
)
from quant_fund.models.kolmogorov_path import (
    bench_kolmogorov_path,
)
from quant_fund.models.path_dependent_pde import (
    bench_path_dependent_pde,
)
from quant_fund.models.path_sobolev import (
    bench_path_sobolev,
)
from quant_fund.models.viscosity_path import (
    bench_viscosity_path,
)


def test_path_dependent_pde():
    assert bench_path_dependent_pde()["synthetic_path_dependent_pde"] == 1.0


def test_functional_ito():
    assert bench_functional_ito()["synthetic_functional_ito"] == 1.0


def test_dupire_functional():
    assert bench_dupire_functional()["synthetic_dupire_functional"] == 1.0


def test_viscosity_path():
    assert bench_viscosity_path()["synthetic_viscosity_path"] == 1.0


def test_path_sobolev():
    assert bench_path_sobolev()["synthetic_path_sobolev"] == 1.0


def test_kolmogorov_path():
    assert bench_kolmogorov_path()["synthetic_kolmogorov_path"] == 1.0
