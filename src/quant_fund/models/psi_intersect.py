"""DH-based private set intersection (OPRF-style, SYNTHETIC bench)."""

from __future__ import annotations

import math
import random

P = 2_147_483_647  # 2^31-1 prime
G = 5


def _unit(rng: random.Random) -> int:
    while True:
        r = rng.randrange(2, P - 2)
        if math.gcd(r, P - 1) == 1:
            return r


def dh_psi(alice: list[int], bob: list[int], rng: random.Random) -> list[int]:
    """Two-message DH PSI: a = g^ka, b = g^kb; compare b^ka vs a^kb sets."""
    ka, kb = rng.randrange(2, P - 2), rng.randrange(2, P - 2)
    a_msgs = [pow(G, x * ka, P) for x in alice]
    b_msgs = [pow(G, x * kb, P) for x in bob]
    a_set = {pow(m, kb, P) for m in a_msgs}
    b_set = {pow(m, ka, P) for m in b_msgs}
    common = a_set & b_set
    # map back: g^{x*ka*kb} shared -> find x in alice ∩ bob via trial
    out = [x for x in alice if pow(G, x * ka * kb, P) in common]
    return sorted(set(out))


def oprf_psi(server_items: list[int], client_items: list[int], rng: random.Random) -> list[int]:
    """OPRF PSI: client blinds x -> g^{x*r}, server answers g^{x*r*k}, unblind to g^{x*k}."""
    k = _unit(rng)
    r = _unit(rng)
    server_set = {pow(G, s * k, P) for s in server_items}
    out = []
    for x in client_items:
        blinded = pow(G, x * r, P)
        resp = pow(blinded, k, P)
        unblinded = pow(resp, pow(r, -1, P - 1), P)
        if unblinded in server_set:
            out.append(x)
    return sorted(set(out))


def _bench_psi_intersect(seed: int = 0) -> float:
    rng = random.Random(20261231 + 1915 + seed)
    checks = []
    alice = [3, 5, 7, 11]
    bob = [5, 7, 13]
    checks.append(dh_psi(alice, bob, rng) == [5, 7])
    checks.append(oprf_psi(alice, bob, rng) == [5, 7])
    checks.append(dh_psi([1, 2], [3, 4], rng) == [])
    checks.append(oprf_psi([1, 2, 3], [2, 3, 4], rng) == [2, 3])
    return sum(checks) / len(checks)


def bench_psi_intersect(seed: int = 0) -> dict[str, float]:
    return {"synthetic_psi_intersect": _bench_psi_intersect(seed)}
