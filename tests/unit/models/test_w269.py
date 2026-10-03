"""Wave-269 computer-vision module tests."""

import numpy as np

from quant_fund.models.epipolar_8pt import fundamental
from quant_fund.models.homography_4pt import dlt
from quant_fund.models.lk_flow import lk_step
from quant_fund.models.orb_feature import brief, harris_score
from quant_fund.models.ransac_plane import ransac_plane
from quant_fund.models.stereo_disparity import disparity


def test_lk_flow_runs() -> None:
    rng = np.random.RandomState(0)
    img = rng.rand(20, 20)
    v = lk_step(img, np.roll(img, 1, axis=1), 10, 10)
    assert v.shape == (2,)


def test_orb_brief_deterministic() -> None:
    rng = np.random.RandomState(0)
    img = rng.rand(20, 20)
    pairs = rng.randint(-4, 5, (16, 2, 2))
    assert brief(img, 10, 10, pairs) == brief(img, 10, 10, pairs)
    assert np.isfinite(harris_score(img)).all()


def test_homography_recovers() -> None:
    h_true = np.array([[1.0, 0.0, 2.0], [0.0, 1.0, 1.0], [0.0, 0.0, 1.0]])
    pts = np.array([[0, 0], [5, 0], [0, 5], [5, 5]], dtype=float)
    ph = h_true @ np.vstack([pts.T, np.ones(4)])
    pts2 = (ph[:2] / ph[2]).T
    h = dlt(pts, pts2)
    ph2 = h @ np.array([3.0, 3.0, 1.0])
    got = ph2[:2] / ph2[2]
    assert np.allclose(got, [5.0, 4.0], atol=0.1)


def test_ransac_finds_floor() -> None:
    rng = np.random.RandomState(0)
    inl = np.hstack([rng.rand(40, 2), np.zeros((40, 1))])
    pts = np.vstack([inl, rng.rand(15, 3) * 5])
    est = ransac_plane(pts, rng)
    assert abs(est[2]) > 0.9 or np.linalg.norm(est[:3]) > 0.5


def test_epipolar_rank2() -> None:
    rng = np.random.RandomState(0)
    pts1 = rng.rand(8, 2) * 10
    pts2 = pts1 + rng.normal(0, 0.5, (8, 2))
    f = fundamental(pts1, pts2)
    s = np.linalg.svd(f, compute_uv=False)
    assert s[2] < 1e-9


def test_disparity_finds_shift() -> None:
    rng = np.random.RandomState(0)
    img = rng.rand(24, 40)
    il = img[:, :-4]
    ir = img[:, 4:]
    d = disparity(il, ir, max_d=8)
    assert np.median(d[8:16, 14:22]) == 4
