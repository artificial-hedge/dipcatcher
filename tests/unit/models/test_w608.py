from quant_fund.models.etale_descent import (
    bench_etale_descent,
)
from quant_fund.models.etale_morphism import (
    bench_etale_morphism,
)
from quant_fund.models.fppf_site import bench_fppf_site
from quant_fund.models.fpqc_site import bench_fpqc_site
from quant_fund.models.ladic_sheaf import bench_ladic_sheaf
from quant_fund.models.lisse_sheaf import bench_lisse_sheaf


def test_etale_descent():
    assert bench_etale_descent()["synthetic_etale_descent"] == 1.0


def test_etale_morphism():
    assert bench_etale_morphism()["synthetic_etale_morphism"] == 1.0


def test_fppf_site():
    assert bench_fppf_site()["synthetic_fppf_site"] == 1.0


def test_fpqc_site():
    assert bench_fpqc_site()["synthetic_fpqc_site"] == 1.0


def test_ladic_sheaf():
    assert bench_ladic_sheaf()["synthetic_ladic_sheaf"] == 1.0


def test_lisse_sheaf():
    assert bench_lisse_sheaf()["synthetic_lisse_sheaf"] == 1.0
