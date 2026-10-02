"""FlashAttention tiled online-softmax (Dao et al. 2022).

Attention computed in row blocks with running (m, l) rescaling — no
materialized T×T score matrix. Verified numerically identical to naive
attention; reports the SRAM-traffic model ratio.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


def _attn_naive(q: FloatArray, k: FloatArray, v: FloatArray) -> FloatArray:
    s = q @ k.T / np.sqrt(q.shape[-1])
    a = np.exp(s - s.max(-1, keepdims=True))
    return np.asarray(a / a.sum(-1, keepdims=True) @ v)


def _attn_flash(q: FloatArray, k: FloatArray, v: FloatArray, block: int = 32) -> FloatArray:
    t, d = q.shape
    out = np.zeros((t, d))
    m = np.full(t, -np.inf)
    ell = np.zeros(t)
    kb = np.ceil(k.shape[0] / block).astype(int)
    for b in range(kb):
        kj = k[b * block : (b + 1) * block]
        vj = v[b * block : (b + 1) * block]
        s = q @ kj.T / np.sqrt(d)
        m_new = np.maximum(m, s.max(-1))
        alpha = np.exp(m - m_new)
        p = np.exp(s - m_new[:, None])
        ell = ell * alpha + p.sum(-1)
        out = out * alpha[:, None] + p @ vj
        m = m_new
    return out / ell[:, None]


def bench_flash_attn(
    seed: int = 157,
    t: int = 256,
    d: int = 32,
    block: int = 32,
) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    q = rng.standard_normal((t, d))
    k = rng.standard_normal((t, d))
    v = rng.standard_normal((t, d))
    ref = _attn_naive(q, k, v)
    got = _attn_flash(q, k, v, block)
    err = float(np.abs(ref - got).max())
    # SRAM traffic model: naive materializes T×T scores twice; tiled reads K,V once per row-block group
    naive_sram = 2 * t * t * 8
    tiled_sram = t * d * 8 + (t // block) * (block * d * 8 * 2)
    return {
        "synthetic_flash_max_err": err,
        "synthetic_flash_sram_ratio": float(tiled_sram) / naive_sram,
        "synthetic_flash_equiv": float(err < 1e-10),
    }
