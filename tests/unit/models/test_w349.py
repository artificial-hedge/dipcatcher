from quant_fund.models.bezout_bezout import bench_bezout_bezout
from quant_fund.models.hilbert_poly import bench_hilbert_poly
from quant_fund.models.monomial_ideal import bench_monomial_ideal
from quant_fund.models.projective_plane import bench_projective_plane
from quant_fund.models.variety_dim import bench_variety_dim
from quant_fund.models.zariski_topo import bench_zariski_topo


def test_zariski_topo():
    assert bench_zariski_topo()["synthetic_zariski_topo"] == 1.0


def test_projective_plane():
    assert bench_projective_plane()["synthetic_projective_plane"] == 1.0


def test_bezout_bezout():
    assert bench_bezout_bezout()["synthetic_bezout_bezout"] == 1.0


def test_variety_dim():
    assert bench_variety_dim()["synthetic_variety_dim"] == 1.0


def test_monomial_ideal():
    assert bench_monomial_ideal()["synthetic_monomial_ideal"] == 1.0


def test_hilbert_poly():
    assert bench_hilbert_poly()["synthetic_hilbert_poly"] == 1.0
