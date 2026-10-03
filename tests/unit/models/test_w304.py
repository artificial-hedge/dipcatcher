"""Unit tests for wave-304 computer-vision-2 canon modules."""

import numpy as np

from quant_fund.models.grabcut_lite import grabcut
from quant_fund.models.harris_corner import detect_corners, harris_response
from quant_fund.models.hough_lines import hough_accumulate, hough_peaks
from quant_fund.models.integral_image import box_sum, integral
from quant_fund.models.meanshift_track import back_project, mean_shift, model_hist, track_step
from quant_fund.models.seam_carving import energy, min_seam, remove_seam


def test_harris_finds_block_corners():
    img = np.zeros((40, 40))
    img[10:30, 10:30] = 1.0
    pts = detect_corners(harris_response(img), 0.0, nms=3)
    assert len(pts) >= 2


def test_hough_recovers_line():
    edge = np.zeros((40, 40))
    edge[20, :] = 1.0
    acc, rhos, thetas = hough_accumulate(edge)
    i, j = hough_peaks(acc, 1, 4)[0]
    assert abs(abs(rhos[i]) - 20) < 2
    assert abs(abs(thetas[j]) - np.pi / 2) < 0.1


def test_integral_matches_brute():
    rng = np.random.default_rng(0)
    img = rng.uniform(0, 1, (20, 20))
    ii = integral(img)
    assert abs(box_sum(ii, 3, 4, 5, 6) - img[3:8, 4:10].sum()) < 1e-9


def test_seam_low_energy_channel():
    img = np.random.default_rng(0).normal(0, 1, (20, 30))
    img[:, 14:18] = 0
    path, _ = min_seam(energy(img))
    assert np.mean(np.abs(np.array(path) - 15.5)) < 3
    assert remove_seam(img, path).shape == (20, 29)


def test_grabcut_iou():
    rng = np.random.default_rng(0)
    img = rng.normal(0.2, 0.05, (40, 40))
    img[10:30, 10:30] = rng.normal(0.85, 0.05, (20, 20))
    p = grabcut(img, (12, 12, 16, 16), iters=4)
    truth = np.zeros((40, 40), bool)
    truth[10:30, 10:30] = True
    iou = ((p > 0.5) & truth).sum() / ((p > 0.5) | truth).sum()
    assert iou > 0.6


def test_meanshift_modes_and_track():
    rng = np.random.default_rng(0)
    pts = np.vstack([rng.normal([0, 0], 0.4, (100, 2)), rng.normal([6, 6], 0.4, (100, 2))])
    assert np.linalg.norm(mean_shift(pts, np.array([0.5, 0.0]), 1.0)) < 0.5
    img = np.zeros((40, 40)) + rng.normal(0.2, 0.05, (40, 40))
    img[5:12, 5:12] = 1.0
    h = model_hist(img[5:12, 5:12], 0.0, 1.0)
    c = track_step(back_project(img, h, 0.0, 1.0), (20.0, 20.0), 12)
    assert abs(c[0] - 8.5) < 4
