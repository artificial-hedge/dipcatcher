from quant_fund.models.analytic_sheaf import (
    bench_analytic_sheaf,
)
from quant_fund.models.clausen_scholze2 import (
    bench_clausen_scholze2,
)
from quant_fund.models.nuclear_space import (
    bench_nuclear_space,
)
from quant_fund.models.proetale_site2 import (
    bench_proetale_site2,
)
from quant_fund.models.solid_cohom import bench_solid_cohom
from quant_fund.models.solid_tensor2 import (
    bench_solid_tensor2,
)


def test_clausen_scholze2():
    assert bench_clausen_scholze2()["synthetic_clausen_scholze2"] == 1.0


def test_solid_cohom():
    assert bench_solid_cohom()["synthetic_solid_cohom"] == 1.0


def test_nuclear_space():
    assert bench_nuclear_space()["synthetic_nuclear_space"] == 1.0


def test_analytic_sheaf():
    assert bench_analytic_sheaf()["synthetic_analytic_sheaf"] == 1.0


def test_solid_tensor2():
    assert bench_solid_tensor2()["synthetic_solid_tensor2"] == 1.0


def test_proetale_site2():
    assert bench_proetale_site2()["synthetic_proetale_site2"] == 1.0
