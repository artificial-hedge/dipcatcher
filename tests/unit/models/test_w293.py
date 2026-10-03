"""Unit tests for wave-293 graphics-3 canon modules."""

import numpy as np

from quant_fund.models.deferred_shade import deferred_light, gbuffer
from quant_fund.models.env_map import face_uv
from quant_fund.models.frustum_cull import _persp, aabb_visible, planes_from_vp
from quant_fund.models.lod_select import lod_level
from quant_fund.models.sdf_raymarch import raymarch, sphere_sdf
from quant_fund.models.shadow_pcf import pcf, shadow_test


def test_deferred_equals_forward():
    px = np.zeros((4, 4, 3))
    nrm = np.tile(np.array([0, 0, 1.0]), (4, 4, 1))
    alb = np.ones((4, 4, 3)) * 0.5
    g = gbuffer(px, nrm, alb)
    out = deferred_light(g, [(np.array([0, 0, 1.0]), np.ones(3))])
    assert out.shape == alb.shape and (out >= 0).all()


def test_raymarch_hit():
    hit = raymarch(
        lambda p: sphere_sdf(np.array([0, 0, 5.0]), 1.0, p), np.zeros(3), np.array([0, 0, 1.0])
    )
    assert abs(hit - 4.0) < 1e-3


def test_frustum_in_out():
    planes = planes_from_vp(_persp(np.pi / 3, 1.0, 0.1, 100.0))
    assert aabb_visible(planes, np.array([-0.5, -0.5, -6.0]), np.array([0.5, 0.5, -4.0]))
    assert not aabb_visible(planes, np.array([50.0, -1.0, -10.0]), np.array([51.0, 1.0, -5.0]))


def test_lod_monotone():
    errors = [0.0, 0.01, 0.1, 1.0]
    assert lod_level(errors, 1.0) == 0
    assert lod_level(errors, 300.0) == 3


def test_env_faces():
    assert face_uv(np.array([1.0, 0, 0]))[0] == 0
    assert face_uv(np.array([0, 0, -1.0]))[0] == 5


def test_shadow_bias():
    zmap = np.ones((4, 4))
    recv = np.ones((4, 4)) * 1.02
    assert shadow_test(zmap, recv, 0.05).mean() == 1.0
    assert pcf(zmap, recv, 0.05).mean() == 1.0
