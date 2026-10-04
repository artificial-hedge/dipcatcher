from quant_fund.models.adjoint_sparse import (
    bench_adjoint_sparse,
)
from quant_fund.models.diffusion_approx_sp import (
    bench_diffusion_approx_sp,
)
from quant_fund.models.element_free import (
    bench_element_free,
)
from quant_fund.models.epi_rk import (
    bench_epi_rk,
)
from quant_fund.models.gauss_rk import (
    bench_gauss_rk,
)
from quant_fund.models.importance_rel import (
    bench_importance_rel,
)


def test_epi_rk():
    assert bench_epi_rk()["synthetic_epi_rk"] == 1.0


def test_gauss_rk():
    assert bench_gauss_rk()["synthetic_gauss_rk"] == 1.0


def test_adjoint_sparse():
    assert bench_adjoint_sparse()["synthetic_adjoint_sparse"] == 1.0


def test_element_free():
    assert bench_element_free()["synthetic_element_free"] == 1.0


def test_diffusion_approx_sp():
    assert bench_diffusion_approx_sp()["synthetic_diffusion_approx_sp"] == 1.0


def test_importance_rel():
    assert bench_importance_rel()["synthetic_importance_rel"] == 1.0
