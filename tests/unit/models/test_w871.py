from quant_fund.models.arnoldi_eig import (
    bench_arnoldi_eig,
)
from quant_fund.models.bicg_solver import (
    bench_bicg_solver,
)
from quant_fund.models.cg_solver import (
    bench_cg_solver,
)
from quant_fund.models.gmres_solver import (
    bench_gmres_solver,
)
from quant_fund.models.lanczos_eig import (
    bench_lanczos_eig,
)
from quant_fund.models.lsqr_solver import (
    bench_lsqr_solver,
)


def test_cg_solver():
    assert bench_cg_solver()["synthetic_cg_solver"] == 1.0


def test_gmres_solver():
    assert bench_gmres_solver()["synthetic_gmres_solver"] == 1.0


def test_bicg_solver():
    assert bench_bicg_solver()["synthetic_bicg_solver"] == 1.0


def test_arnoldi_eig():
    assert bench_arnoldi_eig()["synthetic_arnoldi_eig"] == 1.0


def test_lanczos_eig():
    assert bench_lanczos_eig()["synthetic_lanczos_eig"] == 1.0


def test_lsqr_solver():
    assert bench_lsqr_solver()["synthetic_lsqr_solver"] == 1.0
