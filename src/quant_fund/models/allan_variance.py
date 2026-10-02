"""Overlapping Allan variance / deviation — IMU & clock noise
characterization. Log–log slopes identify white noise (−1/2), flicker
(0), and random walk (+1/2).
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


def allan_deviation(y: FloatArray, dt: float, taus: FloatArray) -> FloatArray:
    """Overlapping Allan deviation at each tau (seconds)."""
    x = np.cumsum(np.asarray(y)) * dt  # integrate rate → phase
    n = x.size
    out = np.empty(len(taus))
    for k, tau in enumerate(taus):
        m = int(round(tau / dt))
        m = max(m, 1)
        if n < 2 * m + 1:
            out[k] = np.nan
            continue
        d = x[2 * m :] - 2 * x[m:-m] + x[: -2 * m]
        out[k] = np.sqrt(np.sum(d * d) / (2 * d.size * (tau * tau)))
    return out


def allan_taus(dt: float, n: int, max_tau_frac: float = 0.25) -> FloatArray:
    """Octave-spaced tau grid up to a fraction of the record length."""
    max_m = int(n * max_tau_frac)
    ms = np.unique(np.clip(np.round(2.0 ** np.arange(0, np.log2(max_m))), 1, max_m).astype(int))
    return ms * dt


def noise_slope(adev: FloatArray, taus: FloatArray) -> float:
    """Log–log slope of Allan deviation vs tau."""
    mask = np.isfinite(adev) & (adev > 0)
    fit = np.polyfit(np.log(taus[mask]), np.log(adev[mask]), 1)
    return float(fit[0])


def bench_allan_variance(seed: int = 20261231) -> dict[str, float]:
    """SYNTHETIC: white noise recovers slope −1/2, integrated white
    noise (random walk) +1/2, at the canonical taus."""
    rng = np.random.default_rng(seed)
    dt = 0.01
    n = 200_000
    out: dict[str, float] = {}
    taus = allan_taus(dt, n)
    white = rng.normal(0, 1.0, n)
    adev_w = allan_deviation(white, dt, taus)
    out["synthetic_allan_white_slope"] = noise_slope(adev_w, taus)
    walk = np.cumsum(white) * 0.01
    adev_r = allan_deviation(walk, dt, taus)
    out["synthetic_allan_walk_slope"] = noise_slope(adev_r, taus)
    out["synthetic_allan_white_ok"] = float(abs(out["synthetic_allan_white_slope"] + 0.5) < 0.06)
    out["synthetic_allan_walk_ok"] = float(abs(out["synthetic_allan_walk_slope"] - 0.5) < 0.08)
    return out


if __name__ == "__main__":
    print(bench_allan_variance())
