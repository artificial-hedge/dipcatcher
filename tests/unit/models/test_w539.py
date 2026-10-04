from quant_fund.models.continued_frac2 import bench_continued_frac2
from quant_fund.models.dirichlet_approx import bench_dirichlet_approx
from quant_fund.models.kronecker_thm import bench_kronecker_thm
from quant_fund.models.liouville_number import bench_liouville_number
from quant_fund.models.roth_thm2 import bench_roth_thm2
from quant_fund.models.subspace_thm import bench_subspace_thm


def test_dirichlet_approx():
    assert bench_dirichlet_approx()["synthetic_dirichlet_approx"] == 1.0


def test_roth_thm2():
    assert bench_roth_thm2()["synthetic_roth_thm2"] == 1.0


def test_continued_frac2():
    assert bench_continued_frac2()["synthetic_continued_frac2"] == 1.0


def test_kronecker_thm():
    assert bench_kronecker_thm()["synthetic_kronecker_thm"] == 1.0


def test_liouville_number():
    assert bench_liouville_number()["synthetic_liouville_number"] == 1.0


def test_subspace_thm():
    assert bench_subspace_thm()["synthetic_subspace_thm"] == 1.0
