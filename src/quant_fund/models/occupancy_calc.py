"""Kernel occupancy: threads-per-SM bound by regs, shared mem, block limits."""

import numpy as np

_SEED = 20261231 + 690


def occupancy(
    threads_per_block: int,
    regs_per_thread: int,
    smem_per_block: int,
    max_threads: int = 2048,
    regs_per_sm: int = 65536,
    smem_per_sm: int = 98304,
) -> float:
    blocks_by_threads = max_threads // threads_per_block
    blocks_by_regs = (
        regs_per_sm // (regs_per_thread * threads_per_block) if regs_per_thread > 0 else 99
    )
    blocks_by_smem = smem_per_sm // smem_per_block if smem_per_block > 0 else 99
    n_blocks = min(blocks_by_threads, blocks_by_regs, blocks_by_smem)
    return min(1.0, n_blocks * threads_per_block / max_threads)


def bench_occupancy_calc(seed: int = _SEED) -> dict[str, float]:
    rng = np.random.RandomState(seed)
    ok = 0.0
    trials = 40
    for _ in range(trials):
        o = occupancy(
            int(rng.choice([32, 64, 128, 256, 512])),
            int(rng.randint(8, 128)),
            int(rng.randint(0, 49152)),
        )
        ok += float(0.0 <= o <= 1.0)
    # known answer: 256 threads, 32 regs, 8K smem -> limited by threads: 8 blocks -> 1.0
    ok += float(occupancy(256, 32, 8192) == 1.0)
    return {"synthetic_occupancy_bounded": ok / (trials + 1)}
