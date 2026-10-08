"""Cubemap env-map honesty tests."""

from __future__ import annotations

import numpy as np
import pytest

from quant_fund.models.env_map import bench_env_map, face_uv


def test_zero_direction_fails_closed():
    with pytest.raises(ValueError, match="zero direction"):
        face_uv(np.zeros(3))


def test_nonfinite_direction_fails_closed():
    with pytest.raises(ValueError, match="finite"):
        face_uv(np.array([np.nan, 0.0, 0.0]))
    with pytest.raises(ValueError, match="finite"):
        face_uv(np.array([np.inf, 0.0, 0.0]))


def test_canonical_faces():
    face, _u, _v = face_uv(np.array([1.0, 0.0, 0.0]))
    assert face == 0
    face, _u, _v = face_uv(np.array([-1.0, 0.0, 0.0]))
    assert face == 1
    face, _u, _v = face_uv(np.array([0.0, 1.0, 0.0]))
    assert face == 2
    face, _u, _v = face_uv(np.array([0.0, -1.0, 0.0]))
    assert face == 3
    face, _u, _v = face_uv(np.array([0.0, 0.0, 1.0]))
    assert face == 4
    face, _u, _v = face_uv(np.array([0.0, 0.0, -1.0]))
    assert face == 5


def test_uv_in_range():
    rng = np.random.RandomState(0)
    for _ in range(200):
        d = rng.normal(size=3)
        _f, u, v = face_uv(d)
        assert 0.0 <= u <= 1.0 and 0.0 <= v <= 1.0


def test_bench_env_map():
    assert bench_env_map()["synthetic_env_map"] == 1.0
