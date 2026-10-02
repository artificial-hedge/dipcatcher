"""Ring attention simulation (Liu et al. 2023).

The K/V sequence is sharded across two virtual devices; each computes
partial online softmax, then a ring merge combines them — output equals
full attention exactly. Reports per-device KV memory fraction and the
equivalence error.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


def _partial(
    q: FloatArray, k: FloatArray, v: FloatArray
) -> tuple[FloatArray, FloatArray, FloatArray]:
    s = q @ k.T / np.sqrt(q.shape[-1])
    m = s.max(-1)
    p = np.exp(s - m[:, None])
    return p @ v, p.sum(-1), m


def bench_ring_attn(
    seed: int = 167,
    t: int = 128,
    d: int = 32,
    n_dev: int = 4,
) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    q = rng.standard_normal((t, d))
    k = rng.standard_normal((t, d))
    v = rng.standard_normal((t, d))
    s_full = q @ k.T / np.sqrt(d)
    a = np.exp(s_full - s_full.max(-1, keepdims=True))
    ref = a / a.sum(-1, keepdims=True) @ v

    chunks = np.array_split(np.arange(t), n_dev)
    out = np.zeros((t, d))
    ell = np.zeros(t)
    m = np.full(t, -np.inf)
    for ch in chunks:
        pj, lj, mj = _partial(q, k[ch], v[ch])
        m_new = np.maximum(m, mj)
        alpha = np.exp(m - m_new)
        out = out * alpha[:, None] + pj * np.exp(mj - m_new)[:, None]
        ell = ell * alpha + lj * np.exp(mj - m_new)
        m = m_new
    got = out / ell[:, None]
    err = float(np.abs(ref - got).max())
    return {
        "synthetic_ring_max_err": err,
        "synthetic_ring_equiv": float(err < 1e-10),
        "synthetic_ring_kv_frac_per_dev": float(1) / n_dev,
        "synthetic_ring_n_dev": float(n_dev),
    }
