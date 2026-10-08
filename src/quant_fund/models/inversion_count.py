"""Merge-sort inversion counting vs quadratic oracle (wave 282) (SYNTHETIC)."""

import numpy as np

_SEED = 20261231 + 779


def inversions(arr: list[int]) -> int:
    if len(arr) <= 1:
        return 0
    mid = len(arr) // 2
    left, right = arr[:mid], arr[mid:]
    cnt = inversions(left) + inversions(right)
    i = j = 0
    merged = []
    while i < len(left) and j < len(right):
        if left[i] <= right[j]:
            merged.append(left[i])
            i += 1
        else:
            merged.append(right[j])
            cnt += len(left) - i
            j += 1
    merged += left[i:] + right[j:]
    arr[:] = merged
    return cnt


def _brute(arr: list[int]) -> int:
    return sum(1 for i in range(len(arr)) for j in range(i + 1, len(arr)) if arr[i] > arr[j])


def bench_inversion_count(seed: int = _SEED) -> dict[str, float]:
    rng = np.random.RandomState(seed)
    ok = 0
    for _ in range(8):
        arr = [int(v) for v in rng.randint(0, 30, int(rng.randint(3, 15)))]
        ok += int(inversions(arr[:]) == _brute(arr))
    # sanity: sorted has 0, reversed has n*(n-1)/2
    ok += int(inversions(list(range(10))) == 0)
    ok += int(inversions(list(range(10, 0, -1))) == 45)
    return {"synthetic_inversion": float(ok == 10)}
