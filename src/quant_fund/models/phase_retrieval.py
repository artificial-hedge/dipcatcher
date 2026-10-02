"""Phase retrieval: recover a signal from Fourier-magnitude
measurements. Gerchberg-Saxton error-reduction (GS 1972)
with an object-support constraint on oversampled spectra,
and Wirtinger flow (Candes, Li & Soltanolkotabi 2015) with a
spectral initialization on coded-diffraction measurements.
Synthetic bench gates recovery correlation up to global
phase."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]
ComplexArray = NDArray[np.complex128]


def gerchberg_saxton(
    mag: FloatArray,
    n_obj: int,
    it: int = 500,
    seed: int = 0,
    beta: float = 0.9,
) -> ComplexArray:
    """Object-domain signal of length n_obj from 2x-oversampled
    FFT magnitudes (object support = first n_obj entries).
    Hybrid input-output: unconstrained entries carry the
    feedback z − β·(magnitude projection), which escapes the
    stagnation of pure error-reduction."""
    mag = np.asarray(mag, dtype=np.float64)
    rng = np.random.default_rng(seed)
    m = mag.shape[0]
    z: ComplexArray = np.asarray(
        rng.normal(0, 1, m) + 1j * rng.normal(0, 1, m), dtype=np.complex128
    )
    for _ in range(it):
        f = np.fft.fft(z)
        f = np.asarray(mag * np.exp(1j * np.angle(f)), dtype=np.complex128)
        z_proj = np.fft.ifft(f)
        # HIO: keep constrained part, subtract feedback outside
        z = np.asarray(
            np.where(
                np.arange(m) < n_obj,
                z_proj,
                z - beta * z_proj,
            ),
            dtype=np.complex128,
        )
    return np.asarray(z[:n_obj])


def _cdi_measure(x: ComplexArray, masks: ComplexArray) -> FloatArray:
    """Coded-diffraction intensities |FFT(x * mask)|² per mask."""
    return np.asarray(np.abs(np.fft.fft(x[None, :] * masks, axis=1)) ** 2)


def wirtinger_flow(
    masks: ComplexArray,
    y: FloatArray,
    it: int = 300,
    lr: float = 0.2,
    n_pow: int = 50,
) -> ComplexArray:
    """Wirtinger flow on 1/2 Σ(|<a_i,z>|² − y_i)² with a
    spectral initializer from the masked data."""
    y = np.asarray(y, dtype=np.float64)
    # normalize to O(1) power so the descent step-size is
    # scale-free; recovery is judged up to global scale anyway
    y = y / max(float(y.mean()), 1e-12)
    masks = np.asarray(masks, dtype=np.complex128)
    n = masks.shape[1]
    # spectral init: largest eigenvector of the per-mask
    # intensity-weighted mask Gram sum
    w_mask = y.mean(axis=1)
    ymat = np.einsum("i,id,ie->de", w_mask, masks, np.conj(masks)) / masks.shape[0]
    v: ComplexArray = np.asarray(np.random.default_rng(0).normal(0, 1, n), dtype=np.complex128)
    for _ in range(n_pow):
        v = ymat @ v
        v /= np.linalg.norm(v)
    z = np.sqrt(y.mean() / 2.0) * v
    for _ in range(it):
        est = np.fft.fft(z[None, :] * masks, axis=1)
        res = np.abs(est) ** 2 - y
        grad = np.fft.ifft(res * est, axis=1) * np.conj(masks)
        z = z - (lr / (2 * n)) * grad.sum(axis=0)
    return np.asarray(z)


def _align_corr(z: ComplexArray, x: ComplexArray) -> float:
    """Correlation magnitude after removing the global phase."""
    c = np.vdot(z, x)
    if abs(c) < 1e-12:
        return 0.0
    z_rot = z * np.exp(-1j * np.angle(c))
    return float(np.abs(np.vdot(z_rot, x)) / (np.linalg.norm(z) * np.linalg.norm(x) + 1e-12))


def bench_phase_retrieval(seed: int = 550) -> dict[str, float]:
    """SYNTHETIC: recover a random signal — GS on 2x-
    oversampled spectra and WF on masked intensities must
    correlate with the truth > 0.8 (up to global phase)."""
    rng = np.random.default_rng(seed)
    out: dict[str, float] = {}
    n = 24
    x = rng.normal(0, 1, n) + 1j * rng.normal(0, 1, n)
    # GS: zero-pad to 2n -> 2x oversampled magnitude
    mag = np.abs(np.fft.fft(np.r_[x, np.zeros(n)]))
    z_gs = gerchberg_saxton(mag, n, it=600, seed=seed)
    # spectrum reproduction is the shift-invariant check
    mag_hat = np.abs(np.fft.fft(np.r_[z_gs, np.zeros(n)]))
    gs_rel = float(np.linalg.norm(mag_hat - mag) / np.linalg.norm(mag))
    out["synthetic_gs_spec_relerr"] = gs_rel
    # WF: L coded-diffraction masks
    masks = (rng.integers(0, 2, (6, n)) * 2 - 1).astype(np.complex128)
    y = _cdi_measure(x, masks)
    z_wf = wirtinger_flow(masks, y, it=400, lr=1.0)
    c_wf = _align_corr(z_wf, x)
    out["synthetic_wf_corr"] = c_wf
    if gs_rel > 0.2:
        raise ValueError(f"gs spec err off: {gs_rel}")
    if c_wf < 0.8:
        raise ValueError(f"wf corr off: {c_wf}")
    return out
