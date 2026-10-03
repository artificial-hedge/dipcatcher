from quant_fund.models.functor_derived import bench_functor_derived
from quant_fund.models.hopf_algebra2 import bench_hopf_algebra2
from quant_fund.models.kunneth import bench_kunneth
from quant_fund.models.leray_hirsch import bench_leray_hirsch
from quant_fund.models.poincare_duality2 import bench_poincare_duality2
from quant_fund.models.universal_coeff import bench_universal_coeff


def test_poincare_duality2():
    assert bench_poincare_duality2()["synthetic_poincare_duality2"] == 1.0


def test_universal_coeff():
    assert bench_universal_coeff()["synthetic_universal_coeff"] == 1.0


def test_kunneth():
    assert bench_kunneth()["synthetic_kunneth"] == 1.0


def test_leray_hirsch():
    assert bench_leray_hirsch()["synthetic_leray_hirsch"] == 1.0


def test_hopf_algebra2():
    assert bench_hopf_algebra2()["synthetic_hopf_algebra2"] == 1.0


def test_functor_derived():
    assert bench_functor_derived()["synthetic_functor_derived"] == 1.0
