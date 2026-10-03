from quant_fund.models.compact_scheme import (
    bench_compact_scheme,
)
from quant_fund.models.crank_nicholson2 import (
    bench_crank_nicholson2,
)
from quant_fund.models.fdm_grid import (
    bench_fdm_grid,
)
from quant_fund.models.flux_splitting import (
    bench_flux_splitting,
)
from quant_fund.models.muscl_reconstruct import (
    bench_muscl_reconstruct,
)
from quant_fund.models.upwind_scheme import (
    bench_upwind_scheme,
)


def test_fdm_grid():
    assert bench_fdm_grid()["synthetic_fdm_grid"] == 1.0


def test_compact_scheme():
    assert bench_compact_scheme()["synthetic_compact_scheme"] == 1.0


def test_crank_nicholson2():
    assert bench_crank_nicholson2()["synthetic_crank_nicholson2"] == 1.0


def test_upwind_scheme():
    assert bench_upwind_scheme()["synthetic_upwind_scheme"] == 1.0


def test_muscl_reconstruct():
    assert bench_muscl_reconstruct()["synthetic_muscl_reconstruct"] == 1.0


def test_flux_splitting():
    assert bench_flux_splitting()["synthetic_flux_splitting"] == 1.0
