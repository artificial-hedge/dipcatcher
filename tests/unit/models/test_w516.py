from quant_fund.models.affine_lie import bench_affine_lie
from quant_fund.models.kac_moody import bench_kac_moody
from quant_fund.models.moonshine_module import bench_moonshine_module
from quant_fund.models.vertex_alg import bench_vertex_alg
from quant_fund.models.weyl_kac import bench_weyl_kac
from quant_fund.models.zhu_algebra import bench_zhu_algebra


def test_kac_moody():
    assert bench_kac_moody()["synthetic_kac_moody"] == 1.0


def test_weyl_kac():
    assert bench_weyl_kac()["synthetic_weyl_kac"] == 1.0


def test_vertex_alg():
    assert bench_vertex_alg()["synthetic_vertex_alg"] == 1.0


def test_moonshine_module():
    assert bench_moonshine_module()["synthetic_moonshine_module"] == 1.0


def test_affine_lie():
    assert bench_affine_lie()["synthetic_affine_lie"] == 1.0


def test_zhu_algebra():
    assert bench_zhu_algebra()["synthetic_zhu_algebra"] == 1.0
