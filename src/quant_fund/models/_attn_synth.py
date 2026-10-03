"""Shared long-range retrieval fixture for wave-136 attention-efficiency benches.

Sequence: [k_1,v_1, k_2,v_2, ..., k_m,v_m, q*] — key tokens embed a class id,
value tokens embed the payload; the query token asks for the value paired with
a specific key. Labels are the target value's class. Attention must match a
key arbitrarily far back — the long-range niche where full O(T^2) attention is
the oracle and efficient mechanisms trade dot-cost for accuracy. SYNTHETIC.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]

_TOK = 8  # embedding dim for token vectors
_NKEYS = 64


def synth_retrieval(
    n: int, m_pairs: int, n_classes: int, rng: np.random.Generator
) -> tuple[FloatArray, NDArray[np.int64]]:
    """x (n, T, d): alternating key/value token embeddings + final query.

    Key tokens: one-hot key id in first _NKEYS slots + noise. Value tokens:
    one-hot class in next slots. Query: last token holds the queried key id.
    """
    d = _NKEYS + n_classes + _TOK
    t = 2 * m_pairs + 1
    x = np.zeros((n, t, d))
    y = np.zeros(n, dtype=np.int64)
    for i in range(n):
        keys = rng.choice(_NKEYS, m_pairs, replace=False)
        vals = rng.integers(0, n_classes, m_pairs)
        for j in range(m_pairs):
            x[i, 2 * j, keys[j]] = 1.0  # key token also carries its value
            x[i, 2 * j, _NKEYS + vals[j]] = 1.0
            x[i, 2 * j + 1, _NKEYS + vals[j]] = 1.0  # distractor value-only token
        q_idx = int(rng.integers(0, m_pairs))
        x[i, -1, keys[q_idx]] = 1.0
        y[i] = vals[q_idx]
        x[i] += 0.05 * rng.standard_normal((t, d))
    return x.astype(np.float64), y


def attn_dot_cost(n_tokens: int, method: str, param: int) -> float:
    """Normalized attention score-computation cost (full attn = 1.0)."""
    full = float(n_tokens * n_tokens)
    if method == "full":
        return 1.0
    if method in ("linformer", "performer", "linear", "nystrom"):
        return float(n_tokens * param) / full
    if method == "sliding":
        return float(n_tokens * param) / full
    if method == "sinkhorn":
        n_blocks = param
        blk = n_tokens // n_blocks
        return float(n_blocks * blk * blk * 2) / full
    if method == "lsh":
        blk = n_tokens // param
        return float(param * blk * blk) / full
    if method == "memknn":
        return float(n_tokens * param) / full
    return 1.0
