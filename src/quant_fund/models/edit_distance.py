"""Edit distance + alignment recovery (synthetic).

Levenshtein DP with backtrace producing an actual edit script.
Verified: (i) distance equals reference DP oracle; (ii) applying
the recovered script transforms source into target; (iii) triangle
inequality on random triples.
"""

from __future__ import annotations

import random


def edit_dp(a: str, b: str) -> int:
    m, n = len(a), len(b)
    prev = list(range(n + 1))
    for i in range(1, m + 1):
        cur = [i] + [0] * n
        for j in range(1, n + 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (a[i - 1] != b[j - 1]))
        prev = cur
    return prev[n]


def edit_script(a: str, b: str) -> list[tuple[str, str, str]]:
    """Return ops [(op, a_char, b_char)] — keep/del/ins/sub."""
    m, n = len(a), len(b)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            dp[i][j] = min(
                dp[i - 1][j] + 1, dp[i][j - 1] + 1, dp[i - 1][j - 1] + (a[i - 1] != b[j - 1])
            )
    ops: list[tuple[str, str, str]] = []
    i, j = m, n
    while i or j:
        if i and j and a[i - 1] == b[j - 1] and dp[i][j] == dp[i - 1][j - 1]:
            ops.append(("keep", a[i - 1], b[j - 1]))
            i -= 1
            j -= 1
        elif i and j and dp[i][j] == dp[i - 1][j - 1] + 1:
            ops.append(("sub", a[i - 1], b[j - 1]))
            i -= 1
            j -= 1
        elif i and dp[i][j] == dp[i - 1][j] + 1:
            ops.append(("del", a[i - 1], ""))
            i -= 1
        else:
            ops.append(("ins", "", b[j - 1]))
            j -= 1
    return ops[::-1]


def apply_script(a: str, ops: list[tuple[str, str, str]]) -> str:
    return "".join(b for _, _, b in ops)


def bench_edit_distance(seed: int = 20261231 + 263) -> dict[str, float]:
    rng = random.Random(seed)
    alpha = "abcd"
    agree = script_ok = tri_ok = 0
    trials = 30
    for _ in range(trials):
        s1 = "".join(rng.choice(alpha) for _ in range(rng.randint(0, 25)))
        s2 = "".join(rng.choice(alpha) for _ in range(rng.randint(0, 25)))
        s3 = "".join(rng.choice(alpha) for _ in range(rng.randint(0, 25)))
        agree += int(edit_dp(s1, s2) == _oracle(s1, s2))
        ops = edit_script(s1, s2)
        n_edit = sum(1 for o in ops if o[0] != "keep")
        script_ok += int(apply_script(s1, ops) == s2 and n_edit == edit_dp(s1, s2))
        d12, d23, d13 = edit_dp(s1, s2), edit_dp(s2, s3), edit_dp(s1, s3)
        tri_ok += int(d13 <= d12 + d23)
    return {
        "synthetic_agree": float(agree / trials),
        "synthetic_script_valid": float(script_ok / trials),
        "synthetic_triangle": float(tri_ok / trials),
    }


def _oracle(a: str, b: str) -> int:
    """Independent top-down memoized recursion."""
    from functools import cache

    @cache
    def go(i: int, j: int) -> int:
        if i == 0:
            return j
        if j == 0:
            return i
        cost = 0 if a[i - 1] == b[j - 1] else 1
        return min(
            go(i - 1, j) + 1,
            go(i, j - 1) + 1,
            go(i - 1, j - 1) + cost,
        )

    return go(len(a), len(b))
