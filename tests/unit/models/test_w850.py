from quant_fund.models.ausm_flux import (
    bench_ausm_flux,
)
from quant_fund.models.godunov_exact import (
    bench_godunov_exact,
)
from quant_fund.models.hllc_solver import (
    bench_hllc_solver,
)
from quant_fund.models.lax_friedrichs import (
    bench_lax_friedrichs,
)
from quant_fund.models.osher_solver import (
    bench_osher_solver,
)
from quant_fund.models.roe_solver import (
    bench_roe_solver,
)


def test_roe_solver():
    assert bench_roe_solver()["synthetic_roe_solver"] == 1.0


def test_hllc_solver():
    assert bench_hllc_solver()["synthetic_hllc_solver"] == 1.0


def test_ausm_flux():
    assert bench_ausm_flux()["synthetic_ausm_flux"] == 1.0


def test_lax_friedrichs():
    assert bench_lax_friedrichs()["synthetic_lax_friedrichs"] == 1.0


def test_godunov_exact():
    assert bench_godunov_exact()["synthetic_godunov_exact"] == 1.0


def test_osher_solver():
    assert bench_osher_solver()["synthetic_osher_solver"] == 1.0
