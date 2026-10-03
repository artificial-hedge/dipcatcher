from quant_fund.models.calabi_yau_tri import (
    bench_calabi_yau_tri,
)
from quant_fund.models.d_calabi_yau import bench_d_calabi_yau
from quant_fund.models.frobenius_cat import bench_frobenius_cat
from quant_fund.models.gorenstein_proj import (
    bench_gorenstein_proj,
)
from quant_fund.models.orbit_category import bench_orbit_category
from quant_fund.models.stable_category import (
    bench_stable_category,
)


def test_calabi_yau_tri():
    assert bench_calabi_yau_tri()["synthetic_calabi_yau_tri"] == 1.0


def test_d_calabi_yau():
    assert bench_d_calabi_yau()["synthetic_d_calabi_yau"] == 1.0


def test_gorenstein_proj():
    assert bench_gorenstein_proj()["synthetic_gorenstein_proj"] == 1.0


def test_frobenius_cat():
    assert bench_frobenius_cat()["synthetic_frobenius_cat"] == 1.0


def test_stable_category():
    assert bench_stable_category()["synthetic_stable_category"] == 1.0


def test_orbit_category():
    assert bench_orbit_category()["synthetic_orbit_category"] == 1.0
