from quant_fund.models.etd_rk4_classic import (
    bench_etd_rk4_classic,
)
from quant_fund.models.expm_int import (
    bench_expm_int,
)
from quant_fund.models.expokit import (
    bench_expokit,
)
from quant_fund.models.krylov_subspace_time import (
    bench_krylov_subspace_time,
)
from quant_fund.models.leja_point import (
    bench_leja_point,
)
from quant_fund.models.phi_function import (
    bench_phi_function,
)


def test_expm_int():
    assert bench_expm_int()["synthetic_expm_int"] == 1.0


def test_expokit():
    assert bench_expokit()["synthetic_expokit"] == 1.0


def test_krylov_subspace_time():
    assert bench_krylov_subspace_time()["synthetic_krylov_subspace_time"] == 1.0


def test_leja_point():
    assert bench_leja_point()["synthetic_leja_point"] == 1.0


def test_phi_function():
    assert bench_phi_function()["synthetic_phi_function"] == 1.0


def test_etd_rk4_classic():
    assert bench_etd_rk4_classic()["synthetic_etd_rk4_classic"] == 1.0
