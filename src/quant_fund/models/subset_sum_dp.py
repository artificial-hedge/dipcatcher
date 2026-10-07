"""Subset-sum dynamic program vs brute-force oracle (wave 282) (SYNTHETIC).

DP table over reachable sums; count witnesses via backtracking-equivalent
path counts. Verified against exhaustive enumeration on random instances.
"""

import itertools

import numpy as np

_SEED = 20261231 + 776


def reachable(nums: list[int], target: int) -> bool:
    ok = {0}
    for v in nums:
        ok |= {s + v for s in ok if s + v <= target}
    return target in ok


def count_ways(nums: list[int], target: int) -> int:
    ways = {0: 1}
    for v in nums:
        for s in sorted(ways.keys(), reverse=True):
            if s + v <= target:
                ways[s + v] = ways.get(s + v, 0) + ways[s]
    return ways.get(target, 0)


def _brute(nums: list[int], target: int) -> int:
    return sum(
        1
        for r in range(len(nums) + 1)
        for comb in itertools.combinations(nums, r)
        if sum(comb) == target
    )


def bench_subset_sum_dp(seed: int = _SEED) -> dict[str, float]:
    rng = np.random.RandomState(seed)
    ok = 0
    for _ in range(8):
        nums = [int(v) for v in rng.randint(1, 12, 8)]
        t = int(rng.randint(1, 40))
        ok += int(reachable(nums, t) == (_brute(nums, t) > 0))
        ok += int(count_ways(nums, t) == _brute(nums, t))
    return {"synthetic_subset_sum": float(ok == 16)}
