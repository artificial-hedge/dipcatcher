"""Synthetic signal fixtures shared by the signal-processing-3 canon.

A linear chirp, a harmonic "voiced" train for cepstrum/LPC, a two-tone
target for Goertzel, and a narrowband 8-sensor array snapshot for MVDR.
"""

import numpy as np

FS = 8000.0
N = 4096


def chirp(seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    t = np.arange(N) / FS
    f = 200.0 + 600.0 * t / (N / FS)
    ph = 2 * np.pi * np.cumsum(f) / FS
    return np.sin(ph) + 0.1 * rng.standard_normal(N)


def voiced(seed: int, f0: float = 120.0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    t = np.arange(N) / FS
    sig = np.zeros(N)
    for k in range(1, 9):
        sig += (1.0 / k) * np.sin(2 * np.pi * k * f0 * t + rng.uniform(0, 2 * np.pi))
    return sig + 0.05 * rng.standard_normal(N)


def two_tone(seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    t = np.arange(N) / FS
    return (
        np.sin(2 * np.pi * 941 * t)
        + 0.5 * np.sin(2 * np.pi * 1336 * t)
        + 0.2 * rng.standard_normal(N)
    )


def array_snap(seed: int, theta_deg: float = 20.0) -> tuple[np.ndarray, np.ndarray]:
    """8-element ULA, half-wavelength spacing; returns (X, steering)."""
    rng = np.random.default_rng(seed)
    m, snaps = 8, 256
    th = np.deg2rad(theta_deg)
    steer = np.exp(-1j * np.pi * np.arange(m) * np.sin(th))
    t = np.arange(snaps)
    s = np.exp(1j * 2 * np.pi * 0.05 * t)
    interf = np.exp(1j * 2 * np.pi * 0.11 * t) * np.exp(
        -1j * np.pi * np.arange(m)[:, None] * np.sin(np.deg2rad(-35.0))
    )
    noise = 0.1 * (rng.standard_normal((m, snaps)) + 1j * rng.standard_normal((m, snaps)))
    x = steer[:, None] * s[None, :] + 0.5 * interf + noise
    return x, steer
