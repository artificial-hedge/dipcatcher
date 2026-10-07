"""Compressed-sensing MRI — undersampled k-space + sparsity-regularized ISTA (SYNTHETIC).

Forward model: k-space samples F x on a random subset of frequencies
(partial Fourier). Recovery: ISTA on ||A x - y||^2 + lambda ||W x||_1
where W is a Haar wavelet basis — soft-thresholding in coefficient space.
Bench: reconstruction PSNR beats the zero-filled inverse FFT.
"""

import numpy as np

_SEED = 20261231 + 874


def _haar_matrix(n: int) -> np.ndarray:
    """Orthonormal Haar wavelet basis for length-n signals (n power of 2)."""
    basis = [np.ones(n) / np.sqrt(n)]
    level = n
    while level > 1:
        level //= 2
        for k in range(n // (2 * level)):
            v = np.zeros(n)
            v[2 * level * k : 2 * level * k + level] = 1.0
            v[2 * level * k + level : 2 * level * k + 2 * level] = -1.0
            basis.append(v / np.linalg.norm(v))
    return np.array(basis)


def _partial_fft(mask: np.ndarray, img: np.ndarray) -> np.ndarray:
    return np.asarray(np.fft.fftshift(np.fft.fft2(img))[mask])


def _adjoint(mask: np.ndarray, y: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    k = np.zeros(shape, dtype=complex)
    k[mask] = y
    return np.asarray(np.fft.ifft2(np.fft.ifftshift(k)).real)


def cs_recon(
    y: np.ndarray, mask: np.ndarray, shape: tuple[int, int], lam: float = 0.02, iters: int = 60
) -> np.ndarray:
    W = _haar_matrix(shape[0] * shape[1])
    x = _adjoint(mask, y, shape)
    step = 1.0
    for _ in range(iters):
        resid = _partial_fft(mask, x) - y
        grad = _adjoint(mask, resid, shape)
        z = (x - step * grad).ravel()
        coeff = W @ z
        coeff = np.sign(coeff) * np.maximum(np.abs(coeff) - lam * step, 0.0)
        x = (W.T @ coeff).reshape(shape)
    return x


def bench_cs_mri(seed: int = _SEED) -> dict[str, float]:
    """SYNTHETIC bench: sparse ISTA PSNR beats zero-filled PSNR."""
    rng = np.random.default_rng(seed)
    n = 32
    img = np.zeros((n, n))
    yy, xx = np.mgrid[0:n, 0:n]
    img[((xx - n / 2) / (n * 0.35)) ** 2 + ((yy - n / 2) / (n * 0.3)) ** 2 <= 1] = 1.0
    img[(xx - n / 2) ** 2 + (yy - n / 4) ** 2 <= (n * 0.1) ** 2] = 0.5
    k = np.fft.fftshift(np.fft.fft2(img))
    keep = rng.random((n, n)) < 0.35
    keep[n // 2 - 2 : n // 2 + 3, n // 2 - 2 : n // 2 + 3] = True
    y = k[keep]
    zf = np.fft.ifft2(np.fft.ifftshift(np.where(keep, k, 0))).real
    rec = cs_recon(y, keep, (n, n))

    def _psnr(a: np.ndarray) -> float:
        return float(-10 * np.log10(np.mean((a - img) ** 2) + 1e-12))

    gain = _psnr(rec) - _psnr(zf)
    return {"synthetic_cs_mri": 1.0 if gain > 1.0 else 0.0}
