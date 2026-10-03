from quant_fund.models.bounded_complex import bench_bounded_complex
from quant_fund.models.derived_functor2 import bench_derived_functor2
from quant_fund.models.koszul_dual import bench_koszul_dual
from quant_fund.models.mapping_cone_tri import bench_mapping_cone_tri
from quant_fund.models.t_structure import bench_t_structure
from quant_fund.models.triangulated import bench_triangulated


def test_derived_functor2():
    assert bench_derived_functor2()["synthetic_derived_functor2"] == 1.0


def test_triangulated():
    assert bench_triangulated()["synthetic_triangulated"] == 1.0


def test_bounded_complex():
    assert bench_bounded_complex()["synthetic_bounded_complex"] == 1.0


def test_mapping_cone_tri():
    assert bench_mapping_cone_tri()["synthetic_mapping_cone_tri"] == 1.0


def test_koszul_dual():
    assert bench_koszul_dual()["synthetic_koszul_dual"] == 1.0


def test_t_structure():
    assert bench_t_structure()["synthetic_t_structure"] == 1.0
