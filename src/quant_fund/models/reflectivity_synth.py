"""Convolutional synthetic seismogram: layered impedance -> reflectivity -> trace.

Reflectivity series r_i = (Z_{i+1} - Z_i) / (Z_{i+1} + Z_i) convolved with a
Ricker wavelet; the canonical zero-phase model of seismic trace generation.
"""

from __future__ import annotations

import numpy as np

_SEED = 20261231 + 932


def impedance_to_reflectivity(z: np.ndarray) -> np.ndarray:
    z = np.asarray(z, dtype=np.float64)
    return np.asarray((z[1:] - z[:-1]) / (z[1:] + z[:-1]))


def ricker(f: float, dt: float, half_len: int | None = None) -> np.ndarray:
    if half_len is None:
        half_len = int(np.ceil(1.5 / (f * dt)))
    t = np.arange(-half_len, half_len + 1) * dt
    a = np.pi**2 * f**2 * t**2
    return (1.0 - 2.0 * a) * np.exp(-a)


def layered_impedance(rng: np.random.Generator, n_layers: int, spacing: int) -> np.ndarray:
    """Piecewise-constant impedance log (sampled)."""
    v = rng.uniform(1800.0, 3200.0, n_layers)
    rho = 0.31 * v**0.25 * rng.uniform(0.98, 1.02, n_layers)
    return np.repeat(v * rho, spacing)


def synthetic_trace(refl: np.ndarray, wavelet: np.ndarray) -> np.ndarray:
    return np.convolve(np.asarray(refl, dtype=np.float64), wavelet)


def _events(refl: np.ndarray) -> np.ndarray:
    return np.flatnonzero(np.abs(refl) > 1e-12)


def _lobe_sign(trace: np.ndarray, idx: int, win: int) -> float:
    lo = max(0, idx - win)
    hi = min(len(trace), idx + win + 1)
    seg = trace[lo:hi]
    j = int(np.argmax(np.abs(seg)))
    return float(np.sign(seg[j]))


def bench_reflectivity_synth(seed: int = _SEED) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    spacing = 60
    z = layered_impedance(rng, n_layers=6, spacing=spacing)
    refl = impedance_to_reflectivity(z)
    dt = 0.001
    w = ricker(30.0, dt)
    trace = synthetic_trace(refl, w)
    truth = np.convolve(refl, w)
    recon = float(np.max(np.abs(trace - truth)))
    ev = _events(refl)
    centre = len(w) // 2
    signs = np.array(
        [_lobe_sign(trace, i + centre, 12) == np.sign(refl[i]) for i in ev],
        dtype=np.float64,
    )
    located = np.array(
        [
            abs(int(np.argmax(np.abs(trace[i + centre - 12 : i + centre + 13]))) - 12) <= 3
            for i in ev
        ],
        dtype=np.float64,
    )
    return {
        "synthetic_reflectivity_synth": float(
            0.4 * signs.mean()
            + 0.3 * located.mean()
            + 0.2 * (recon < 1e-10)
            + 0.1 * (np.max(np.abs(refl)) < 1.0)
        )
    }
