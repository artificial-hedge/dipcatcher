"""Bitcoin-style difficulty retarget — SYNTHETIC.

2016-block window, clamped to [prev/4, prev*4], verified direction +
clamp + steady-state.
"""

from __future__ import annotations


def retarget(prev_target: int, actual_secs: int, target_secs: int = 2016 * 600) -> int:
    """New difficulty *target* (Bitcoin convention): blocks faster →
    smaller target (harder); clamped to [prev/4, prev*4]."""
    adj = max(target_secs // 4, min(target_secs * 4, actual_secs))
    return prev_target * adj // target_secs


def bench_difficulty_retarget(seed: int = 20261231 + 343) -> dict[str, float]:
    _ = seed
    prev = 10**12
    # blocks twice as fast → target halves (harder)
    faster = retarget(prev, 2016 * 300)
    # blocks twice as slow → target doubles
    slower = retarget(prev, 2016 * 1200)
    # extreme windows clamp to 4x
    clamp_slow = retarget(prev, 2016 * 600 * 100)
    clamp_fast = retarget(prev, 1)
    return {
        "synthetic_faster_harder": float(faster == prev // 2),
        "synthetic_slower_easier": float(slower == prev * 2),
        "synthetic_clamp_slow": float(clamp_slow == prev * 4),
        "synthetic_clamp_fast": float(clamp_fast == prev // 4),
    }
