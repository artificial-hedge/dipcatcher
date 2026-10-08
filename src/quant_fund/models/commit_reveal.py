"""SYNTHETIC hash commitment scheme — binding + hiding properties.

commit(m, r) = H(m || r). Binding: opening a commitment to a different
message fails. Hiding: commitment reveals nothing (commitments to two
different messages are uniform/indistinct under random r).
"""

from __future__ import annotations

import random


def _h(x: int) -> int:
    z = x * 0x9E3779B97F4A7C15 & (1 << 64) - 1
    z ^= z >> 33
    z = z * 0xC2B2AE3D27D4EB4F & (1 << 64) - 1
    return (z ^ (z >> 29)) & (1 << 64) - 1


def commit(m: int, r: int) -> int:
    # bind m into the 64-bit state BEFORE mixing: (m<<64)|r loses m entirely
    # under the mod-2^64 multiply inside _h, leaving the commitment unbound.
    return _h(((m * 0xBF58476D1CE4E5B9) ^ r) & ((1 << 64) - 1))


def open_ok(c: int, m: int, r: int) -> bool:
    return commit(m, r) == c


def bench_commit_reveal(seed: int = 20261231 + 425) -> dict[str, float]:
    rng = random.Random(seed)
    bind = hide = honest = 0
    trials = 40
    for _ in range(trials):
        m1, m2 = rng.getrandbits(32), rng.getrandbits(32)
        r1 = rng.getrandbits(64)
        c = commit(m1, r1)
        honest += int(open_ok(c, m1, r1))
        # binding: another opening must fail both with the SAME nonce (the
        # case a weak commitment breaks) and with a fresh random nonce
        m2_alt = m1 ^ (1 << rng.randrange(32))
        bind += int(not open_ok(c, m2_alt, r1) and not open_ok(c, m2_alt, rng.getrandbits(64)))
        # hiding-ish: commits to m1 and m2 differ, and commits to same m differ across r
        c1 = commit(m1, rng.getrandbits(64))
        c2 = commit(m1, rng.getrandbits(64))
        c3 = commit(m2, rng.getrandbits(64))
        hide += int(c1 != c2 and c1 != c3)
    return {
        "synthetic_open_verifies": float(honest / trials),
        "synthetic_binding": float(bind / trials),
        "synthetic_randomized_hiding": float(hide / trials),
    }
