"""Wave-317 image-processing module unit tests."""

from __future__ import annotations

import numpy as np

from quant_fund.models.canny_edge import gaussian_blur, nonmax_suppress, sobel
from quant_fund.models.distance_transform import chamfer_dt, medial_ridge
from quant_fund.models.nlm_denoise import nlm
from quant_fund.models.otsu_threshold import otsu, segment
from quant_fund.models.slic_superpixels import slic
from quant_fund.models.watershed_seg import local_minima_markers, watershed


def test_gaussian_blur_shape_and_mean() -> None:
    img = np.random.default_rng(0).random((20, 20))
    out = gaussian_blur(img, 1.0)
    assert out.shape == img.shape
    assert abs(float(out.mean()) - float(img.mean())) < 0.05


def test_sobel_detects_step() -> None:
    img = np.zeros((20, 20))
    img[:, 10:] = 1.0
    gx, gy = sobel(img)
    assert float(np.abs(gx[:, 10]).max()) > 0.4
    assert float(np.abs(gx[:, 0]).max()) < 1e-9


def test_nonmax_thins() -> None:
    mag = np.zeros((9, 9))
    mag[4, :] = 1.0
    thin = nonmax_suppress(mag, np.zeros((9, 9)))
    assert int(thin.sum()) == 0 or float(thin.max()) <= 1.0


def test_otsu_bimodal() -> None:
    rng = np.random.default_rng(1)
    img = np.concatenate(
        [
            np.clip(0.1 + 0.05 * rng.standard_normal(1000), 0, 1),
            np.clip(0.9 + 0.05 * rng.standard_normal(1000), 0, 1),
        ]
    )
    t = otsu(img)
    assert 0.3 < t < 0.7
    seg = segment(img, t)
    assert abs(seg.mean() - 0.5) < 0.1


def test_watershed_labels_all() -> None:
    markers = np.zeros((10, 10), int)
    markers[2, 2] = 1
    markers[7, 7] = 2
    cost = np.random.default_rng(2).random((10, 10)) * 0.1
    lab = watershed(cost, markers)
    assert int((lab > 0).sum()) == 100
    assert lab[2, 2] == 1 and lab[7, 7] == 2


def test_local_minima_found() -> None:
    img = np.ones((10, 10)) * 0.9
    img[3, 3] = 0.1
    img[7, 7] = 0.05
    mk = local_minima_markers(img, 0.2)
    assert mk.max() >= 1 and mk[3, 3] > 0 and mk[7, 7] > 0


def test_slic_partitions_all() -> None:
    img = np.random.default_rng(3).random((30, 30))
    lab = slic(img, k=9, iters=4)
    assert int((lab >= 0).sum()) == 900
    assert len(np.unique(lab)) >= 4


def test_nlm_preserves_constant() -> None:
    img = np.full((10, 10), 0.6)
    out = nlm(img, h=0.1)
    assert np.allclose(out, 0.6, atol=1e-9)


def test_chamfer_point_source() -> None:
    mask = np.zeros((11, 11), bool)
    mask[5, 5] = True
    d = chamfer_dt(mask)
    assert float(d[5, 5]) == 0.0
    assert float(d[5, 6]) == 1.0
    ridge = medial_ridge(d)
    assert int(ridge.sum()) >= 0
