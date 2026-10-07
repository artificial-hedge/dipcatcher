"""Adversarial probes for _vision_synth (SYNTHETIC)."""

from __future__ import annotations

import numpy as np
import pytest

from quant_fund.models._vision_synth import patches, synth_images


def test_synth_images_deterministic():
    x1, y1 = synth_images(0, n=32)
    x2, y2 = synth_images(0, n=32)
    np.testing.assert_array_equal(x1, x2)
    np.testing.assert_array_equal(y1, y2)


def test_synth_images_xor_structure():
    x, y = synth_images(1, n=400, noise=0.05)
    tl = x[:, 0:2, 0:2].mean(axis=(1, 2)) > 0.7
    br = x[:, 4:6, 4:6].mean(axis=(1, 2)) > 0.7
    np.testing.assert_array_equal(y, (tl ^ br).astype(np.int64))


def test_synth_images_not_linearly_separable():
    x, y = synth_images(2, n=400)
    flat = x.reshape(400, -1)
    w = np.linalg.lstsq(np.hstack([flat, np.ones((400, 1))]), y.astype(float), rcond=None)[0]
    pred = (np.hstack([flat, np.ones((400, 1))]) @ w > 0.5).astype(int)
    assert (pred == y).mean() < 0.85  # XOR caps linear near chance-ish


@pytest.mark.parametrize("n,noise", [(0, 0.3), (-1, 0.3), (8, -0.1), (8, np.nan)])
def test_synth_images_hostile(n, noise):
    with pytest.raises(ValueError):
        synth_images(0, n=n, noise=noise)


def test_patches_shape_and_content():
    x, _ = synth_images(0, n=4)
    p2 = patches(x, 2)
    assert p2.shape == (4, 9, 4)
    np.testing.assert_array_equal(p2[:, 0], x[:, 0:2, 0:2].reshape(4, 4))
    p3 = patches(x, 3)
    assert p3.shape == (4, 4, 9)
    p6 = patches(x, 6)
    assert p6.shape == (4, 1, 36)


@pytest.mark.parametrize("p", [0, -2, 4, 5, 7])
def test_patches_bad_p(p):
    x, _ = synth_images(0, n=4)
    with pytest.raises(ValueError):
        patches(x, p)


def test_patches_hostile_x():
    x, _ = synth_images(0, n=4)
    with pytest.raises(ValueError):
        patches(x[:0], 2)  # empty
    with pytest.raises(ValueError):
        patches(np.zeros((4, 5, 6)), 2)  # wrong spatial dims
    with pytest.raises(ValueError):
        patches(np.zeros((4, 6)), 2)  # not 3-D
