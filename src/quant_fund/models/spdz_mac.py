"""SPDZ-style information-theoretic MACs on additive shares (SYNTHETIC bench)."""

from __future__ import annotations

import random


def share_maced(
    v: int, alpha: list[int], n: int, p: int, rng: random.Random
) -> tuple[list[int], list[int]]:
    """Share v additively and v*alpha additively: (x_i, m_i) per party."""
    xs = [rng.randrange(p) for _ in range(n - 1)]
    xs.append((v - sum(xs)) % p)
    mac_val = v * (sum(alpha) % p) % p
    ms = [rng.randrange(p) for _ in range(n - 1)]
    ms.append((mac_val - sum(ms)) % p)
    return xs, ms


def open_check(xs: list[int], ms: list[int], alpha: list[int], p: int) -> tuple[bool, int]:
    """Open x = sum xs; check sum ms == alpha*x where alpha = sum alpha_i."""
    x = sum(xs) % p
    a = sum(alpha) % p
    return (sum(ms) % p) == (a * x % p), x


def add(
    a: tuple[list[int], list[int]], b: tuple[list[int], list[int]], p: int
) -> tuple[list[int], list[int]]:
    return (
        [(x + y) % p for x, y in zip(a[0], b[0], strict=True)],
        [(x + y) % p for x, y in zip(a[1], b[1], strict=True)],
    )


def _bench_spdz_mac(seed: int = 0) -> float:
    rng = random.Random(20261231 + 1914 + seed)
    p = 251
    n = 3
    checks = []
    alpha = [rng.randrange(p) for _ in range(n)]
    xs, ms = share_maced(50, alpha, n, p, rng)
    ok, v = open_check(xs, ms, alpha, p)
    checks.append(ok and v == 50)
    # add two maced values: (x+y) verifies
    xs2, ms2 = share_maced(30, alpha, n, p, rng)
    xs3, ms3 = add((xs, ms), (xs2, ms2), p)
    ok3, v3 = open_check(xs3, ms3, alpha, p)
    checks.append(ok3 and v3 == 80)
    # tampering detection: adversary flips a share
    xs_bad = list(xs)
    xs_bad[0] = (xs_bad[0] + 1) % p
    ok_bad, _ = open_check(xs_bad, ms, alpha, p)
    checks.append(not ok_bad)
    return sum(checks) / len(checks)


def bench_spdz_mac(seed: int = 0) -> dict[str, float]:
    return {"synthetic_spdz_mac": _bench_spdz_mac(seed)}
