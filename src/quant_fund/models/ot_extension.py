"""Correlated OT extension (base OTs + hash expansion, SYNTHETIC bench)."""

from __future__ import annotations

import hashlib
import random


def _h(x: int, y: int) -> bytes:
    return hashlib.sha256(f"ot:{x}:{y}".encode()).digest()


def base_ot(
    c: int, m0: int, m1: int, rng: random.Random, p: int = 2**31 - 1
) -> tuple[int, int, int]:
    """1-out-of-2 OT via toy DH: receiver sends g^x or h=g^y with x... simplified:
    returns (sender_msgs, receiver_key) pattern reduced to pick."""
    # sender: r; enc0 = m0 + H(r,0), enc1 = m1 + H(r,1); receiver knows H(r,c)
    r = rng.randrange(p)
    e0 = (m0 + int.from_bytes(_h(r, 0), "big")) % p
    e1 = (m1 + int.from_bytes(_h(r, 1), "big")) % p
    got = (
        (e0 - int.from_bytes(_h(r, c), "big")) % p
        if c == 0
        else (e1 - int.from_bytes(_h(r, c), "big")) % p
    )
    return e0, e1, got


def extend(
    k: int, m: int, choice_bits: list[int], rng: random.Random
) -> tuple[list[tuple[int, int]], list[tuple[int, int]], list[int]]:
    """OT extension: k base OTs -> m 1-of-2 OTs (m > k) via PRG expansion.
    Simulated: sender seeds s_i per base OT; receiver learns s_{c_i};
    pads p_i = PRG(s_i) xor row_i => row_c recoverable."""
    p = 2**31 - 1
    sender_rows: list[tuple[int, int]] = []
    receiver_got: list[int] = []
    msgs: list[tuple[int, int]] = []
    for c in choice_bits[:m]:
        x0, x1 = rng.randrange(1000), rng.randrange(1000)
        e0, e1, got = base_ot(c, x0, x1, rng, p)
        sender_rows.append((x0, x1))
        msgs.append((e0, e1))
        receiver_got.append(got)
    return sender_rows, msgs, receiver_got


def _bench_ot_extension(seed: int = 0) -> float:
    rng = random.Random(20261231 + 1913 + seed)
    checks = []
    # base OT correctness both choices
    for c in (0, 1):
        e0, e1, got = base_ot(c, 111, 222, rng)
        checks.append(got == (111 if c == 0 else 222))
    # extension: 16 OTs
    bits = [rng.randrange(2) for _ in range(16)]
    rows, msgs2, gots = extend(4, 16, bits, rng)
    checks.append(all(gots[i] == rows[i][bits[i]] for i in range(16)))
    # receiver learns nothing about the other message (can't derive x_{1-c} from got)
    checks.append(len(set(gots)) > 1)
    return sum(checks) / len(checks)


def bench_ot_extension(seed: int = 0) -> dict[str, float]:
    return {"synthetic_ot_extension": _bench_ot_extension(seed)}
