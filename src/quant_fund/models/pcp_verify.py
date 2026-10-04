"""PCP-flavored proof check: Hadamard encoding + BLR linearity test (SYNTHETIC bench)."""

from __future__ import annotations

import random


def hadamard(a: list[int]) -> dict[int, int]:
    """Encode a in {0,1}^n as f(x) = a·x mod 2 for all x in {0,1}^n."""
    n = len(a)
    table: dict[int, int] = {}
    for x in range(1 << n):
        acc = 0
        for i in range(n):
            if (x >> i) & 1:
                acc ^= a[i]
        table[x] = acc
    return table


def blr_test(f: dict[int, int], n: int, trials: int, rng: random.Random) -> int:
    """Count linearity failures f(x)^f(y) != f(x^y) over random pairs."""
    fails = 0
    for _ in range(trials):
        x = rng.randrange(1 << n)
        y = rng.randrange(1 << n)
        if f.get(x, 0) ^ f.get(y, 0) != f.get(x ^ y, 0):
            fails += 1
    return fails


def decode(f: dict[int, int], n: int, trials: int, rng: random.Random) -> list[int]:
    """Self-correction decode: a_i = f(x ^ e_i) ^ f(x) majority vote."""
    out = []
    for i in range(n):
        ei = 1 << i
        votes = [0, 0]
        for _ in range(trials):
            x = rng.randrange(1 << n)
            votes[f.get(x ^ ei, 0) ^ f.get(x, 0)] += 1
        out.append(0 if votes[0] > votes[1] else 1)
    return out


def verify(f: dict[int, int], n: int, rng: random.Random) -> bool:
    """Accept iff table passes BLR and self-correction re-encodes consistently."""
    if blr_test(f, n, 40, rng) > 0:
        return False
    a = decode(f, n, 30, rng)
    return blr_test(f, n, 40, rng) == 0 and decode(f, n, 30, rng) == a


def _bench_pcp_verify(seed: int = 0) -> float:
    rng = random.Random(20261231 + 1069)
    checks = []
    a = [1, 0, 1]
    f = hadamard(a)
    checks.append(blr_test(f, 3, 30, rng) == 0)
    checks.append(decode(f, 3, 40, rng) == a)
    checks.append(verify(f, 3, rng))
    bad = dict(f)
    bad[3] ^= 1  # corrupt one entry
    dec = decode(bad, 3, 40, rng)
    checks.append(dec == a)  # self-correction tolerates 1 bad entry
    # a fully corrupted/random table fails BLR w.h.p.
    rnd_tbl = {x: rng.randrange(2) for x in range(1 << 3)}
    checks.append(blr_test(rnd_tbl, 3, 30, rng) > 0)
    return sum(checks) / len(checks)


def bench_pcp_verify(seed: int = 0) -> dict[str, float]:
    return {"synthetic_pcp_verify": _bench_pcp_verify(seed)}
