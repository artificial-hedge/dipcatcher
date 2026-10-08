"""Stirling numbers of the second kind and Bell numbers (wave 282) (SYNTHETIC).

S(n,k) = k*S(n-1,k) + S(n-1,k-1); B(n) = sum_k S(n,k). Inclusion-exclusion
oracle: S(n,k) = (1/k!) sum_j (-1)^j C(k,j) (k-j)^n.
"""

import math

_SEED = 20261231 + 777


def stirling2(n: int, k: int) -> int:
    dp = [0] * (k + 1)
    dp[0] = 1
    for i in range(1, n + 1):
        for j in range(min(i, k), 0, -1):
            dp[j] = j * dp[j] + dp[j - 1]
        dp[0] = 0
    return dp[k]


def _oracle(n: int, k: int) -> int:
    return int(
        sum((-1) ** j * math.comb(k, j) * (k - j) ** n for j in range(k + 1)) // math.factorial(k)
    )


def bell(n: int) -> int:
    return sum(stirling2(n, k) for k in range(n + 1))


def bench_stirling_count(seed: int = _SEED) -> dict[str, float]:
    ok = all(stirling2(n, k) == _oracle(n, k) for n in range(1, 10) for k in range(1, n + 1))
    bells = [bell(n) for n in range(8)]
    want = [1, 1, 2, 5, 15, 52, 203, 877]
    return {"synthetic_stirling": float(ok and bells == want)}
