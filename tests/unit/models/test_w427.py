from quant_fund.models.bsd_toy import bench_bsd_toy
from quant_fund.models.elliptic_height import bench_elliptic_height
from quant_fund.models.lseries_toy import bench_lseries_toy
from quant_fund.models.modularity_toy import bench_modularity_toy
from quant_fund.models.mordell_weil import bench_mordell_weil
from quant_fund.models.padic_integral import bench_padic_integral


def test_elliptic_height():
    assert bench_elliptic_height()["synthetic_elliptic_height"] == 1.0


def test_mordell_weil():
    assert bench_mordell_weil()["synthetic_mordell_weil"] == 1.0


def test_lseries_toy():
    assert bench_lseries_toy()["synthetic_lseries_toy"] == 1.0


def test_bsd_toy():
    assert bench_bsd_toy()["synthetic_bsd_toy"] == 1.0


def test_modularity_toy():
    assert bench_modularity_toy()["synthetic_modularity_toy"] == 1.0


def test_padic_integral():
    assert bench_padic_integral()["synthetic_padic_integral"] == 1.0
