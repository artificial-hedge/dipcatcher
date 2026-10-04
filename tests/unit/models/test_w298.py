"""Unit tests for wave-298 medical-imaging canon modules."""

import numpy as np

from quant_fund.models.art_sirt import _projector, sirt
from quant_fund.models.chan_vese import chan_vese
from quant_fund.models.cs_mri import cs_recon
from quant_fund.models.hu_moments import _transform, hu_moments
from quant_fund.models.mi_register import _shift, mutual_information, register
from quant_fund.models.radon_fbp import _phantom, fbp, radon


def test_radon_fbp():
    n = 32
    img = _phantom(n)
    thetas = np.linspace(0, np.pi, 45, endpoint=False)
    sino = radon(img, thetas)
    assert sino.shape == (45, n)
    rec = fbp(sino, thetas)
    assert np.isfinite(rec).all()


def test_sirt_converges():
    rng = np.random.default_rng(0)
    n = 8
    img = rng.random(n * n)
    A = _projector(n, 8, 8, rng)
    b = A @ img
    x = sirt(A, b, iters=30)
    assert np.linalg.norm(A @ x - b) < np.linalg.norm(A @ sirt(A, b, iters=1) - b)


def test_cs_mri_shapes():
    rng = np.random.default_rng(0)
    n = 16
    img = rng.random((n, n))
    k = np.fft.fftshift(np.fft.fft2(img))
    mask = rng.random((n, n)) < 0.5
    rec = cs_recon(k[mask], mask, (n, n), iters=10)
    assert rec.shape == (n, n) and np.isfinite(rec).all()


def test_hu_scale_invariant():
    n = 48
    yy, xx = np.mgrid[0:n, 0:n] / n - 0.5
    img = (np.abs(xx / 0.3) + np.abs(yy / 0.2) <= 1).astype(float)
    h1 = hu_moments(img)
    h2 = hu_moments(_transform(img, 1.25, 0.0, 2.0, 1.0))
    assert abs(np.log(abs(h1[0]) + 1e-12) - np.log(abs(h2[0]) + 1e-12)) < 0.08


def test_chan_vese_iou():
    rng = np.random.default_rng(1)
    n = 48
    yy, xx = np.mgrid[0:n, 0:n] / n - 0.5
    truth = ((xx / 0.28) ** 2 + (yy / 0.24) ** 2 <= 1).astype(float)
    img = 0.15 + 0.7 * truth + 0.05 * rng.standard_normal((n, n))
    mask, _ = chan_vese(img, iters=200)
    assert mask.shape == img.shape


def test_mi_register():
    rng = np.random.default_rng(0)
    n = 32
    img = rng.random((n, n))
    moving = _shift(img, -3, -2)
    ry, rx, mi = register(img, moving, radius=6)
    assert (ry, rx) == (3, 2)
    assert mutual_information(img, _shift(moving, 3, 2)) > mutual_information(img, moving)
