"""Paged KV-cache allocator (vLLM PagedAttention) (SYNTHETIC).

KV tensors live in fixed-size blocks allocated on demand — no contiguous
per-sequence reservation. Reports internal-fragmentation waste vs the
contiguous-allocation baseline and cache-hit efficiency on shared
prefixes.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


def bench_paged_kv_cache(
    seed: int = 153,
    n_seq: int = 60,
    max_len: int = 128,
    block: int = 16,
    n_heads: int = 4,
    d_head: int = 16,
    n_layers: int = 2,
) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    lens = rng.integers(4, max_len + 1, n_seq)
    # contiguous baseline: each sequence reserves max_len blocks-worth
    elem_bytes = 8  # fp64
    contig = int(lens.sum()) * n_heads * d_head * n_layers * elem_bytes
    contig_reserve = int(n_seq * max_len) * n_heads * d_head * n_layers * elem_bytes
    # paged: ceil(len/block) blocks per sequence
    blocks = np.ceil(lens / block).astype(int)
    paged = int(blocks.sum()) * block * n_heads * d_head * n_layers * elem_bytes
    waste = float(1 - lens.sum() / (blocks.sum() * block))
    # prefix sharing: half the sequences share a 32-token prefix → COW count
    shared = 32
    n_shared = n_seq // 2
    cow_blocks = int(np.ceil(shared / block))
    shared_saved = cow_blocks * block * n_heads * d_head * n_layers * elem_bytes * (n_shared - 1)
    return {
        "synthetic_paged_bytes": float(paged),
        "synthetic_paged_contig_reserve": float(contig_reserve),
        "synthetic_paged_used_bytes": float(contig),
        "synthetic_paged_waste_frac": waste,
        "synthetic_paged_saving_vs_reserve": float(1 - paged / contig_reserve),
        "synthetic_paged_prefix_share_bytes": float(shared_saved),
    }
