from quant_fund.models.cohomology_cup import bench_cohomology_cup
from quant_fund.models.elliptic_curve import bench_elliptic_curve
from quant_fund.models.koszul_complex import bench_koszul_complex
from quant_fund.models.mayer_vietoris import bench_mayer_vietoris
from quant_fund.models.p_adic_val import bench_p_adic_val
from quant_fund.models.quadratic_recip import bench_quadratic_recip


def test_quadratic_recip():
    assert bench_quadratic_recip()["synthetic_quadratic_recip"] == 1.0


def test_elliptic_curve():
    assert bench_elliptic_curve()["synthetic_elliptic_curve"] == 1.0


def test_p_adic_val():
    assert bench_p_adic_val()["synthetic_p_adic_val"] == 1.0


def test_cohomology_cup():
    assert bench_cohomology_cup()["synthetic_cohomology_cup"] == 1.0


def test_koszul_complex():
    assert bench_koszul_complex()["synthetic_koszul_complex"] == 1.0


def test_mayer_vietoris():
    assert bench_mayer_vietoris()["synthetic_mayer_vietoris"] == 1.0
