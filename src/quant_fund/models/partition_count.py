"""Integer partitions p(n) (SYNTHETIC)."""

from __future__ import annotations


def p_n(n: int) -> int:
    """p(n) via DP: number of partitions of n."""
    dp = [1] + [0] * n
    for part in range(1, n + 1):
        for s in range(part, n + 1):
            dp[s] += dp[s - part]
    return dp[n]


def p_nk(n: int, k: int) -> int:
    """Partitions of n into exactly k parts (DP)."""
    dp = [[0] * (k + 1) for _ in range(n + 1)]
    dp[0][0] = 1
    for i in range(1, n + 1):
        for j in range(1, k + 1):
            dp[i][j] = dp[i - 1][j - 1] + (dp[i - j][j] if i >= j else 0)
    return dp[n][k]


def _bench_partition_count(seed: int = 0) -> float:
    checks = []
    checks.append(p_n(10) == 42)
    checks.append(p_n(5) == 7)
    checks.append(p_n(0) == 1)
    # partitions of 6 into 2 parts = 3 (5+1,4+2,3+3)
    checks.append(p_nk(6, 2) == 3)
    # p(n,1) = p(n,n) = 1
    checks.append(p_nk(7, 1) == 1 and p_nk(7, 7) == 1)
    # row sums over k give p(n)
    checks.append(sum(p_nk(6, k) for k in range(1, 7)) == p_n(6))
    return float(sum(checks) / len(checks))


def bench_partition_count(seed: int = 0) -> dict[str, float]:
    return {"synthetic_partition_count": _bench_partition_count(seed)}
