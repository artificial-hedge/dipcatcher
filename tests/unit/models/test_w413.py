from quant_fund.models.abelian_ext import bench_abelian_ext
from quant_fund.models.artin_lemma import bench_artin_lemma
from quant_fund.models.frobenius_el import bench_frobenius_el
from quant_fund.models.inseparable import bench_inseparable
from quant_fund.models.kummer_ext import bench_kummer_ext
from quant_fund.models.normal_basis import bench_normal_basis


def test_artin_lemma():
    assert bench_artin_lemma()["synthetic_artin_lemma"] == 1.0


def test_normal_basis():
    assert bench_normal_basis()["synthetic_normal_basis"] == 1.0


def test_kummer_ext():
    assert bench_kummer_ext()["synthetic_kummer_ext"] == 1.0


def test_abelian_ext():
    assert bench_abelian_ext()["synthetic_abelian_ext"] == 1.0


def test_frobenius_el():
    assert bench_frobenius_el()["synthetic_frobenius_el"] == 1.0


def test_inseparable():
    assert bench_inseparable()["synthetic_inseparable"] == 1.0
