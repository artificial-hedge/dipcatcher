"""Wave-266 graphics-2 module tests."""

import numpy as np

from quant_fund.models.bump_map import perturb_normal
from quant_fund.models.mipmap_sample import trilinear
from quant_fund.models.phong_shade import phong
from quant_fund.models.shadow_map import shadow_pass
from quant_fund.models.ssao_lite import ssao
from quant_fund.models.triangle_raster import rasterize


def test_raster_covers_centroid_triangle() -> None:
    v0 = np.array([5.0, 5.0, 0.5])
    v1 = np.array([25.0, 5.0, 0.5])
    v2 = np.array([15.0, 25.0, 0.5])
    img = rasterize(v0, v1, v2)
    assert img[:, :, 0].sum() > 100
    assert img[15, 15, 0] == 1.0
    assert img[2, 2, 0] == 0.0


def test_raster_winding_invariant() -> None:
    v0 = np.array([5.0, 5.0, 0.5])
    v1 = np.array([25.0, 5.0, 0.5])
    v2 = np.array([15.0, 25.0, 0.5])
    a = rasterize(v0, v1, v2)
    b = rasterize(v0, v2, v1)
    assert np.array_equal(a[:, :, 0], b[:, :, 0])


def test_raster_depth_interp() -> None:
    v0 = np.array([5.0, 5.0, 0.0])
    v1 = np.array([25.0, 5.0, 1.0])
    v2 = np.array([15.0, 25.0, 0.5])
    img = rasterize(v0, v1, v2)
    covered = img[:, :, 0] > 0
    assert img[:, :, 1][covered].min() >= 0.0
    assert img[:, :, 1][covered].max() <= 1.0


def test_phong_facing_light() -> None:
    n = np.array([0.0, 0.0, 1.0])
    assert phong(n, n, np.array([0.0, 0.0, 1.0])) > 0.9


def test_phong_back_face_dark() -> None:
    n = np.array([0.0, 0.0, 1.0])
    light = np.array([0.0, 0.0, -1.0])
    assert phong(n, light, np.array([0.0, 0.0, 1.0])) <= 0.11


def test_trilinear_midpoint() -> None:
    tex = np.zeros((8, 8))
    v = trilinear(tex, 0.5, 0.5, 0.0)
    assert v == 0.0
    tex2 = np.ones((8, 8))
    assert trilinear(tex2, 0.3, 0.7, 1.5) == 1.0


def test_shadow_lit_vs_occluded() -> None:
    depth = np.full((16, 16), 0.5)
    lit_frag = np.array([[0.0, 0.0, 0.4]])
    dark_frag = np.array([[0.0, 0.0, 0.9]])
    view = np.eye(4)
    assert shadow_pass(depth, lit_frag, view)[0]
    assert not shadow_pass(depth, dark_frag, view)[0]


def test_bump_flat_gives_up() -> None:
    n = perturb_normal(np.zeros((8, 8)), 0.5, 0.5)
    assert np.allclose(n, [0, 0, 1])


def test_bump_tilted() -> None:
    h = np.zeros((8, 8))
    h[4, 5] = 1.0
    n = perturb_normal(h, 0.5, 0.5)
    assert n[2] < 1.0


def test_ssao_range() -> None:
    rng = np.random.RandomState(0)
    depth = rng.rand(16, 16)
    ao = ssao(depth, rng)
    assert ao.min() >= 0.0 and ao.max() <= 1.0
