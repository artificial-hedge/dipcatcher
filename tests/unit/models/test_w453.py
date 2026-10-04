from quant_fund.models.bar_cobar import bench_bar_cobar
from quant_fund.models.deligne_conj import bench_deligne_conj
from quant_fund.models.factor_homology import bench_factor_homology
from quant_fund.models.hochschild_hom import bench_hochschild_hom
from quant_fund.models.operad_koszul import bench_operad_koszul
from quant_fund.models.primitive_elts import bench_primitive_elts


def test_operad_koszul():
    assert bench_operad_koszul()["synthetic_operad_koszul"] == 1.0


def test_bar_cobar():
    assert bench_bar_cobar()["synthetic_bar_cobar"] == 1.0


def test_factor_homology():
    assert bench_factor_homology()["synthetic_factor_homology"] == 1.0


def test_hochschild_hom():
    assert bench_hochschild_hom()["synthetic_hochschild_hom"] == 1.0


def test_deligne_conj():
    assert bench_deligne_conj()["synthetic_deligne_conj"] == 1.0


def test_primitive_elts():
    assert bench_primitive_elts()["synthetic_primitive_elts"] == 1.0
