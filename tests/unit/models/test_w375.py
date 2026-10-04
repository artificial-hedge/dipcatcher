from quant_fund.models.blowup import bench_blowup
from quant_fund.models.elliptic_group import bench_elliptic_group
from quant_fund.models.moduli_stable import bench_moduli_stable
from quant_fund.models.riemann_roch import bench_riemann_roch
from quant_fund.models.scheme_local import bench_scheme_local
from quant_fund.models.sheaf_cohomology import bench_sheaf_cohomology


def test_riemann_roch():
    assert bench_riemann_roch()["synthetic_riemann_roch"] == 1.0


def test_sheaf_cohomology():
    assert bench_sheaf_cohomology()["synthetic_sheaf_cohomology"] == 1.0


def test_scheme_local():
    assert bench_scheme_local()["synthetic_scheme_local"] == 1.0


def test_blowup():
    assert bench_blowup()["synthetic_blowup"] == 1.0


def test_elliptic_group():
    assert bench_elliptic_group()["synthetic_elliptic_group"] == 1.0


def test_moduli_stable():
    assert bench_moduli_stable()["synthetic_moduli_stable"] == 1.0
