from quant_fund.models.galerkin_projection import (
    bench_galerkin_projection,
)
from quant_fund.models.periodic_spline import (
    bench_periodic_spline,
)
from quant_fund.models.polyharmonic_rbf import (
    bench_polyharmonic_rbf,
)
from quant_fund.models.thin_plate_spline import (
    bench_thin_plate_spline,
)
from quant_fund.models.trefethen_diff import (
    bench_trefethen_diff,
)
from quant_fund.models.zernike_poly import (
    bench_zernike_poly,
)


def test_thin_plate_spline():
    assert bench_thin_plate_spline()["synthetic_thin_plate_spline"] == 1.0


def test_polyharmonic_rbf():
    assert bench_polyharmonic_rbf()["synthetic_polyharmonic_rbf"] == 1.0


def test_trefethen_diff():
    assert bench_trefethen_diff()["synthetic_trefethen_diff"] == 1.0


def test_galerkin_projection():
    assert bench_galerkin_projection()["synthetic_galerkin_projection"] == 1.0


def test_periodic_spline():
    assert bench_periodic_spline()["synthetic_periodic_spline"] == 1.0


def test_zernike_poly():
    assert bench_zernike_poly()["synthetic_zernike_poly"] == 1.0
