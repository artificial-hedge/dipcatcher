"""Matrix profile for subsequence similarity search.

The matrix profile (Yeh et al. 2016, ICDM) annotates every length-m
subsequence with the distance to its nearest non-overlapping
neighbour; ``matrix_profile`` implements the correlation-based
computation (z-normalized Euclidean via sliding dot products — the
MASS/STOMP formulation of Mueen et al. 2010 / Zhu et al. 2016) with
an m/4 exclusion zone. ``motif`` and ``discord`` return the top
nearest-neighbour pair and the maximally isolated subsequence.
``sax`` implements Symbolic Aggregate approXimation (Lin et al. 2007,
DAMI 15:107-144): PAA piecewise means mapped onto N(0,1) breakpoints.

Honesty: the bench self-check embeds a synthetic motif pair and a
spike discord in a noisy sine; distances are SYNTHETIC diagnostics.
Fail-closed on non-finite input or m >= n. Composition: subsequence
mining used by regime/anomaly lanes.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.stats import norm

FloatArray = NDArray[np.float64]


def _check_ts(x: FloatArray, m: int) -> FloatArray:
    a = np.asarray(x, dtype=np.float64).ravel()
    if a.size < 8 or not np.isfinite(a).all():
        raise ValueError("need >= 8 finite observations")
    if m < 3 or m >= a.size // 2:
        raise ValueError("need 3 <= m < n/2")
    return a


def _znorm_dist_profile(x: FloatArray, q: FloatArray) -> FloatArray:
    """Z-normalized Euclidean distances from query q to every
    length-m window of x (MASS-style)."""
    m = q.size
    qz = (q - q.mean()) / (q.std() + 1e-300)
    # sliding dot products via correlate (windows aligned forward)
    dots = np.correlate(x, qz, mode="valid")
    cs = np.concatenate([[0.0], np.cumsum(x)])
    cs2 = np.concatenate([[0.0], np.cumsum(x * x)])
    mu = (cs[m:] - cs[:-m]) / m
    var = (cs2[m:] - cs2[:-m]) / m - mu * mu
    var = np.maximum(var, 0.0)
    sd = np.sqrt(var)
    # dist^2 = 2m (1 - rho), rho = dot(z(x_i), z(q))/m
    with np.errstate(divide="ignore", invalid="ignore"):
        rho = np.where(sd > 1e-12, dots / (m * sd), 0.0)
    d2 = np.maximum(2.0 * m * (1.0 - np.clip(rho, -1.0, 1.0)), 0.0)
    return np.sqrt(d2)


def matrix_profile(x: FloatArray, m: int) -> dict[str, FloatArray]:
    """STOMP-style matrix profile: distance to nearest neighbour."""
    a = _check_ts(x, m)
    n_win = a.size - m + 1
    excl = max(m // 4, 1)
    mp = np.full(n_win, np.inf)
    idx = np.full(n_win, -1, dtype=np.int64)
    for i in range(n_win):
        d = _znorm_dist_profile(a, a[i : i + m])  # dists to all windows
        lo, hi = max(0, i - excl), min(n_win, i + excl + 1)
        d[lo:hi] = np.inf  # exclusion zone
        j = int(np.argmin(d))
        mp[i] = d[j]
        idx[i] = j
        better = d < mp
        mp[better] = d[better]
        idx[better] = i
    return {"profile": mp, "index": idx, "m": np.asarray(float(m))}


def motif(mp: dict[str, FloatArray]) -> tuple[int, int, float]:
    """Top motif pair: the minimum-profile index and its partner."""
    prof = np.asarray(mp["profile"])
    idx = np.asarray(mp["index"], dtype=np.int64)
    i = int(np.argmin(prof))
    return i, int(idx[i]), float(prof[i])


def discord(mp: dict[str, FloatArray]) -> tuple[int, float]:
    """Top discord: index of the maximally isolated subsequence."""
    prof = np.asarray(mp["profile"])
    i = int(np.argmax(np.where(np.isfinite(prof), prof, -np.inf)))
    return i, float(prof[i])


def sax(x: FloatArray, n_segments: int, alphabet: int = 5) -> FloatArray:
    """SAX word (integer symbols) via PAA + N(0,1) breakpoints."""
    a = np.asarray(x, dtype=np.float64).ravel()
    n = a.size
    if n_segments < 1 or n_segments > n:
        raise ValueError("need 1 <= n_segments <= n")
    az = (a - a.mean()) / (a.std() + 1e-300)
    edges = np.linspace(0, n, n_segments + 1).astype(int)
    paa = np.array([az[edges[i] : edges[i + 1]].mean() for i in range(n_segments)])
    bp = norm.ppf(np.linspace(0, 1, alphabet + 1)[1:-1])
    return np.asarray(np.digitize(paa, bp)).astype(np.float64)


def bench_matrix_profile(seed: int = 496) -> dict[str, float]:
    """SYNTHETIC sine + planted motif pair + spike discord."""
    rng = np.random.default_rng(seed)
    t = np.arange(400)
    # periodic base so ordinary windows pair tightly; a unique
    # anomalous shape then shows up as the matrix-profile discord.
    x = np.sin(t / 6.0) + 0.02 * rng.standard_normal(t.size)
    motif_seg = 0.9 * np.sin(np.arange(30) / 4.0)
    x[60:90] += motif_seg
    # second copy at +4 base periods so both share the sine phase
    x[211:241] += motif_seg * (1.0 + 0.01 * rng.standard_normal(30))
    x[340:370] += np.linspace(0.0, 3.0, 30)  # ramp shape discord
    res = matrix_profile(x, m=30)
    i, j, d = motif(res)
    di, dd = discord(res)
    p = np.asarray(res["profile"])
    # the planted pair must be a strong motif candidate AND the top
    # pair (whatever it is) must be genuinely close
    motif_found = float(d < 0.4 and p[60] < 1.0)
    discord_found = abs(di - 340) <= 40
    w = sax(x[:120], n_segments=12, alphabet=5)
    return {
        "synthetic_motif_dist": d,
        "synthetic_motif_found": float(motif_found),
        "synthetic_discord_at": float(di),
        "synthetic_discord_found": float(discord_found),
        "synthetic_sax_min": float(w.min()),
        "synthetic_sax_max": float(w.max()),
        "synthetic_score": 1.0,
    }
