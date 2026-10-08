"""Banded Needleman-Wunsch vs full-matrix oracle (wave 284) (SYNTHETIC).

For sequences differing by few edits, a width-w band around the diagonal
contains the optimal alignment — DP restricted to |i-j| <= w matches full NW.
"""

import numpy as np

_SEED = 20261231 + 791


def _nw(s: str, t: str) -> int:
    m, n = len(s), len(t)
    dp = np.zeros((m + 1, n + 1), dtype=int)
    dp[0, :] = np.arange(n + 1) * -1
    dp[:, 0] = np.arange(m + 1) * -1
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            dp[i, j] = max(
                dp[i - 1, j] - 1,
                dp[i, j - 1] - 1,
                dp[i - 1, j - 1] + (1 if s[i - 1] == t[j - 1] else -1),
            )
    return int(dp[m, n])


def _nw_band(s: str, t: str, w: int) -> int:
    m, n = len(s), len(t)
    neg = -(10**9)
    dp = np.full((m + 1, n + 1), neg, dtype=int)
    dp[0, :] = -np.arange(n + 1)
    dp[:, 0] = -np.arange(m + 1)
    for i in range(1, m + 1):
        for j in range(max(1, i - w), min(n, i + w) + 1):
            dp[i, j] = max(
                dp[i - 1, j] - 1,
                dp[i, j - 1] - 1,
                dp[i - 1, j - 1] + (1 if s[i - 1] == t[j - 1] else -1),
            )
    return int(dp[m, n])


def bench_band_align(seed: int = _SEED) -> dict[str, float]:
    rng = np.random.RandomState(seed)
    alpha = "ACGT"
    ok = 0
    for _ in range(8):
        s = "".join(rng.choice(list(alpha), 24))
        tl = list(s)
        for _ in range(int(rng.randint(1, 4))):  # few point mutations
            tl[int(rng.randint(0, 24))] = alpha[int(rng.randint(0, 4))]
        t = "".join(tl)
        ok += int(_nw_band(s, t, 4) == _nw(s, t))
    return {"synthetic_band_align": float(ok == 8)}
