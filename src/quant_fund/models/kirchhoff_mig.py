"""Kirchhoff (diffraction-stack) poststack time migration (SYNTHETIC).

Exploding reflector model: image(x, z) += trace(x_r, t = 2*dist/v).
A point diffractor appears as a hyperbola in the unmigrated section
and focuses at its apex after migration.
"""

from __future__ import annotations

import numpy as np

_SEED = 20261231 + 892


def diffractor_gather(x: np.ndarray, v: float, xd: float, zd: float) -> np.ndarray:
    """Zero-offset arrival times of a point diffractor at (xd, zd)."""
    x = np.asarray(x, dtype=np.float64)
    return 2.0 * np.sqrt((x - xd) ** 2 + zd * zd) / v


def kirchhoff_migrate(data: np.ndarray, x: np.ndarray, t: np.ndarray, v: float) -> np.ndarray:
    """Kirchhoff migration of a zero-offset section onto the same (x,t) grid."""
    d = np.asarray(data, dtype=np.float64)
    x = np.asarray(x, dtype=np.float64)
    t = np.asarray(t, dtype=np.float64)
    dt = float(t[1] - t[0])
    nt = t.size
    img = np.zeros_like(d)
    for iz, z in enumerate(v * t / 2.0):  # image depth row per output time
        for ix, xi in enumerate(x):
            acc = 0.0
            for k, xr in enumerate(x):
                tt = 2.0 * np.sqrt((xr - xi) ** 2 + z * z) / v
                it = int(round(tt / dt))
                if 0 <= it < nt:
                    acc += d[it, k]
            img[iz, ix] = acc
    return img


def bench_kirchhoff_mig(seed: int = _SEED) -> dict[str, float]:
    nt, nx = 300, 40
    dt, dx = 0.004, 25.0
    t = np.arange(nt) * dt
    x = np.arange(nx) * dx
    v = 2000.0
    xd, zd = 500.0, 300.0
    data = np.zeros((nt, nx))
    times = diffractor_gather(x, v, xd, zd)
    for k in range(nx):
        it = int(round(times[k] / dt))
        if 0 <= it < nt:
            data[it, k] = 1.0
    img = kirchhoff_migrate(data, x, t, v)
    score = 0.0
    iz, ix = np.unravel_index(np.argmax(img), img.shape)
    # apex should land within ~2 samples of true diffractor
    loc_err = np.sqrt((x[ix] - xd) ** 2 + (v * t[iz] / 2 - zd) ** 2)
    score += 1.0 if loc_err < 2.0 * dx else 0.0
    # focusing: peak/mean energy ratio higher in migrated than input
    foc_in = float(np.max(np.abs(data)) / (np.mean(np.abs(data)) + 1e-12))
    foc_out = float(np.max(np.abs(img)) / (np.mean(np.abs(img)) + 1e-12))
    score += 1.0 if foc_out > foc_in else 0.0
    # flat-reflector sanity: a horizontal event should stay put
    flat = np.zeros((nt, nx))
    it0 = int(round(2 * 200.0 / v / dt))
    flat[it0, :] = 1.0
    img2 = kirchhoff_migrate(flat, x, t, v)
    col = np.argmax(img2[:, nx // 2])
    score += 1.0 if abs(col - it0) <= 2 else 0.0
    return {"synthetic_kirchhoff_mig": score / 3.0}
