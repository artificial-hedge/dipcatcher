from quant_fund.models.elliptic_surface import bench_elliptic_surface
from quant_fund.models.kodaira_fiber import bench_kodaira_fiber
from quant_fund.models.mordell_weil2 import bench_mordell_weil2
from quant_fund.models.neron_model import bench_neron_model
from quant_fund.models.tate_algorithm import bench_tate_algorithm
from quant_fund.models.weierstrass_eq import bench_weierstrass_eq


def test_elliptic_surface():
    assert bench_elliptic_surface()["synthetic_elliptic_surface"] == 1.0


def test_weierstrass_eq():
    assert bench_weierstrass_eq()["synthetic_weierstrass_eq"] == 1.0


def test_kodaira_fiber():
    assert bench_kodaira_fiber()["synthetic_kodaira_fiber"] == 1.0


def test_tate_algorithm():
    assert bench_tate_algorithm()["synthetic_tate_algorithm"] == 1.0


def test_mordell_weil2():
    assert bench_mordell_weil2()["synthetic_mordell_weil2"] == 1.0


def test_neron_model():
    assert bench_neron_model()["synthetic_neron_model"] == 1.0
