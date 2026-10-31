"""Numba kernels for stationary-bootstrap index fill and count tables.

Bit-identical to the pure-NumPy reference when the same Bernoulli mask and
origin stream are supplied. Optional: without numba the callers keep their
interpreted loops.
"""

from __future__ import annotations

from typing import Any

import numpy as np

try:
    from numba import njit

    HAVE_NUMBA = True
except (ImportError, OSError):  # pragma: no cover - numba-less / broken llvmlite
    HAVE_NUMBA = False

    def njit(*_a: Any, **_k: Any) -> Any:  # type: ignore[no-redef]
        def _deco(fn: Any) -> Any:
            return fn

        if len(_a) == 1 and callable(_a[0]) and not _k:
            return _a[0]
        return _deco


@njit(cache=True)
def fill_stationary_indices(
    out: np.ndarray,
    new_block: np.ndarray,
    origins_flat: np.ndarray,
    origin_offsets: np.ndarray,
    n: int,
) -> None:
    """Fill ``out[b, j]`` from block-start origins (Politis–Romano geometry)."""
    n_boot = out.shape[0]
    for b in range(n_boot):
        off = origin_offsets[b]
        origin = 0
        start = 0
        oi = 0
        for j in range(n):
            if j == 0 or new_block[b, j]:
                origin = origins_flat[off + oi]
                oi += 1
                start = j
                out[b, j] = origin
            else:
                out[b, j] = (origin + (j - start)) % n


@njit(cache=True)
def fill_bootstrap_counts(counts: np.ndarray, idx: np.ndarray) -> None:
    """``counts[b, t] = #{positions in idx[b] equal to t}``."""
    n_boot, n = idx.shape
    for b in range(n_boot):
        for t in range(n):
            counts[b, idx[b, t]] += 1.0
