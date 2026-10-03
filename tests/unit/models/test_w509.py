from quant_fund.models.deligne_weil2 import bench_deligne_weil2
from quant_fund.models.etale_site2 import bench_etale_site2
from quant_fund.models.frobenius_action import bench_frobenius_action
from quant_fund.models.groth_lefschetz import bench_groth_lefschetz
from quant_fund.models.l_adic_sheaf import bench_l_adic_sheaf
from quant_fund.models.purity_thm import bench_purity_thm


def test_etale_site2():
    assert bench_etale_site2()["synthetic_etale_site2"] == 1.0


def test_l_adic_sheaf():
    assert bench_l_adic_sheaf()["synthetic_l_adic_sheaf"] == 1.0


def test_frobenius_action():
    assert bench_frobenius_action()["synthetic_frobenius_action"] == 1.0


def test_groth_lefschetz():
    assert bench_groth_lefschetz()["synthetic_groth_lefschetz"] == 1.0


def test_deligne_weil2():
    assert bench_deligne_weil2()["synthetic_deligne_weil2"] == 1.0


def test_purity_thm():
    assert bench_purity_thm()["synthetic_purity_thm"] == 1.0
