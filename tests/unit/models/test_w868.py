from quant_fund.models.arc_continuation import (
    bench_arc_continuation,
)
from quant_fund.models.bifurcation_track import (
    bench_bifurcation_track,
)
from quant_fund.models.davidenko_ode import (
    bench_davidenko_ode,
)
from quant_fund.models.deflation_method import (
    bench_deflation_method,
)
from quant_fund.models.homotopy_solver import (
    bench_homotopy_solver,
)
from quant_fund.models.pseudo_arclength import (
    bench_pseudo_arclength,
)


def test_arc_continuation():
    assert bench_arc_continuation()["synthetic_arc_continuation"] == 1.0


def test_pseudo_arclength():
    assert bench_pseudo_arclength()["synthetic_pseudo_arclength"] == 1.0


def test_deflation_method():
    assert bench_deflation_method()["synthetic_deflation_method"] == 1.0


def test_bifurcation_track():
    assert bench_bifurcation_track()["synthetic_bifurcation_track"] == 1.0


def test_homotopy_solver():
    assert bench_homotopy_solver()["synthetic_homotopy_solver"] == 1.0


def test_davidenko_ode():
    assert bench_davidenko_ode()["synthetic_davidenko_ode"] == 1.0
