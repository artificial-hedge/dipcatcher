"""Fold every corpus byte into a fixed-width integer state.

Each byte updates exactly one lane (index modulo the lane count) through an
odd multiply-add. That step is a bijection on uint64, so changing any byte
changes its lane, including bytes that share a lane with later bytes.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

# Odd constant, so multiplication is a bijection on uint64.
_HASH_K = np.uint64(1315423911)
_GOLDEN = np.uint64(0x9E3779B97F4A7C15)


def fold_lanes(tokens: NDArray[np.uint8], n_lanes: int, seed: int) -> NDArray[np.uint64]:
    """Return ``n_lanes`` uint64 accumulators that depend on every token."""
    if n_lanes < 1:
        raise ValueError("n_lanes must be positive")
    if tokens.ndim != 1:
        raise ValueError("tokens must be a 1-D byte array")
    if int(tokens.dtype.itemsize) != 1:
        raise ValueError("tokens must be a byte array")
    lanes = _seed_lanes(n_lanes, seed)
    n = int(tokens.shape[0])
    cursor = 0
    while cursor < n:
        take = min(n_lanes, n - cursor)
        raw = np.asarray(tokens[cursor : cursor + take], dtype=np.uint64)
        lanes[:take] = lanes[:take] * _HASH_K + raw + np.uint64(1)
        cursor += take
    return lanes


def lanes_to_unit_interval(lanes: NDArray[np.uint64]) -> NDArray[np.float64]:
    """Map lanes to ``[0, 1]`` using the low 24 bits.

    A one-byte change leaves an odd difference in the lane value, so those
    low bits move and the unit value changes.
    """
    mixed = (lanes & np.uint64(0xFFFFFF)).astype(np.float64)
    return mixed / float(0xFFFFFF)


def _seed_lanes(n_lanes: int, seed: int) -> NDArray[np.uint64]:
    idx = np.arange(n_lanes, dtype=np.uint64)
    seed_u = np.uint64(seed % (2**32))
    return (seed_u + np.uint64(1)) * (idx * _GOLDEN + np.uint64(1))
