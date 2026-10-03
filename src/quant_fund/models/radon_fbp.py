"""Radon transform + filtered back-projection — tomographic reconstruction.

Forward model: sinogram(theta, s) = line integrals of the image along rays
at angle theta, computed by rotating the image with nearest-neighbour
resampling and summing columns. Reconstruction: ramp-filter each sinogram
row in Fourier space, then back-project. Bench compares FBP output to the
phantom by correlation.
"""

import numpy as np

_SEED = 20261231 + 872


def _rotate(img: np.ndarray, theta: float) -> np.ndarray:
    n = img.shape[0]
    c = (n - 1) / 2.0
    co, si = np.cos(theta), np.sin(theta)
    yy, xx = np.mgrid[0:n, 0:n]
    xs = co * (xx - c) + si * (yy - c) + c
    ys = -si * (xx - c) + co * (yy - c) + c
    xi, yi = np.round(xs).astype(int), np.round(ys).astype(int)
    valid = (xi >= 0) & (xi < n) & (yi >= 0) & (yi < n)
    out = np.zeros_like(img)
    out[valid] = img[yi[valid], xi[valid]]
    return out


def radon(img: np.ndarray, thetas: np.ndarray) -> np.ndarray:
    return np.asarray(np.stack([_rotate(img, t).sum(axis=0) for t in thetas]))


def fbp(sino: np.ndarray, thetas: np.ndarray) -> np.ndarray:
    """Ramp-filtered back-projection."""
    n_th, n_s = sino.shape
    n_out = n_s
    freqs = np.abs(np.fft.rfftfreq(n_s) * n_s)
    filt_sino = np.fft.irfft(np.fft.rfft(sino, axis=1) * freqs[None, :], n=n_s, axis=1)
    img = np.zeros((n_out, n_out))
    for i, t in enumerate(thetas):
        back = np.tile(filt_sino[i], (n_out, 1))
        img += _rotate(back, -t)
    return np.asarray(img * np.pi / (2 * n_th))


def _phantom(n: int) -> np.ndarray:
    yy, xx = np.mgrid[0:n, 0:n] / n - 0.5
    img = np.zeros((n, n))
    img[(xx / 0.4) ** 2 + (yy / 0.35) ** 2 <= 1] = 0.6
    img[(xx / 0.15) ** 2 + (yy / 0.2) ** 2 <= 1] = 1.0
    img[((xx - 0.15) / 0.08) ** 2 + ((yy + 0.1) / 0.12) ** 2 <= 1] = 0.2
    return img


def bench_radon_fbp(seed: int = _SEED) -> dict[str, float]:
    """SYNTHETIC bench: FBP reconstruction correlates with the ground-truth phantom."""
    rng = np.random.default_rng(seed)
    n = 48
    img = _phantom(n)
    thetas = rng.uniform(0, np.pi, 60)
    sino = radon(img, thetas)
    rec = fbp(sino, thetas)
    rec = rec[:n, :n] if rec.shape[0] >= n else rec
    a, b = rec.ravel() - rec.mean(), img.ravel() - img.mean()
    corr = float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))
    # Line-integral consistency: sinogram mass == image mass per angle (approx).
    mass_err = abs(sino[0].sum() - img.sum()) / img.sum()
    return {"synthetic_radon_fbp": 1.0 if (corr > 0.85 and mass_err < 0.05) else corr}
