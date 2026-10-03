from quant_fund.models.abel_jacobi import bench_abel_jacobi
from quant_fund.models.branched_cover import bench_branched_cover
from quant_fund.models.fuchsian_group import bench_fuchsian_group
from quant_fund.models.riemann_hurwitz import bench_riemann_hurwitz
from quant_fund.models.riemann_surface import bench_riemann_surface
from quant_fund.models.teichmuller_space import bench_teichmuller_space


def test_riemann_surface():
    assert bench_riemann_surface()["synthetic_riemann_surface"] == 1.0


def test_branched_cover():
    assert bench_branched_cover()["synthetic_branched_cover"] == 1.0


def test_abel_jacobi():
    assert bench_abel_jacobi()["synthetic_abel_jacobi"] == 1.0


def test_riemann_hurwitz():
    assert bench_riemann_hurwitz()["synthetic_riemann_hurwitz"] == 1.0


def test_fuchsian_group():
    assert bench_fuchsian_group()["synthetic_fuchsian_group"] == 1.0


def test_teichmuller_space():
    assert bench_teichmuller_space()["synthetic_teichmuller_space"] == 1.0
