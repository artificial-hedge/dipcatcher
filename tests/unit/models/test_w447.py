from quant_fund.models.bord_cat import bench_bord_cat
from quant_fund.models.chern_simons import bench_chern_simons
from quant_fund.models.dw_theory import bench_dw_theory
from quant_fund.models.extended_tqft import bench_extended_tqft
from quant_fund.models.frobenius_2d import bench_frobenius_2d
from quant_fund.models.tqft_axiom import bench_tqft_axiom


def test_tqft_axiom():
    assert bench_tqft_axiom()["synthetic_tqft_axiom"] == 1.0


def test_bord_cat():
    assert bench_bord_cat()["synthetic_bord_cat"] == 1.0


def test_frobenius_2d():
    assert bench_frobenius_2d()["synthetic_frobenius_2d"] == 1.0


def test_extended_tqft():
    assert bench_extended_tqft()["synthetic_extended_tqft"] == 1.0


def test_dw_theory():
    assert bench_dw_theory()["synthetic_dw_theory"] == 1.0


def test_chern_simons():
    assert bench_chern_simons()["synthetic_chern_simons"] == 1.0
