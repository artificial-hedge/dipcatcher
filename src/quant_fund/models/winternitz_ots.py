"""SYNTHETIC Winternitz one-time signature scheme (W-OTS).

Chains of w=4 over a 16-nibble digest with a toy hash; signature reveals
chain elements, verifier finishes chains to the public key. Forge attempt
(one nibble flipped) must fail.
"""

from __future__ import annotations

import random

_W = 4
_N = 16  # nibbles in digest


def _h(rng_state: int, x: int) -> int:
    # deterministic toy "hash": 64-bit mix
    z = (x ^ rng_state) * 0x9E3779B97F4A7C15 & (1 << 64) - 1
    z ^= z >> 33
    z = z * 0xC2B2AE3D27D4EB4F & (1 << 64) - 1
    return z ^ (z >> 29)


def _chain(salt: int, x: int, steps: int) -> int:
    for _ in range(steps):
        x = _h(salt, x)
    return x


def keygen(rng: random.Random) -> tuple[list[int], list[int], int]:
    salt = rng.getrandbits(64)
    sk = [rng.getrandbits(64) for _ in range(_N)]
    pk = [_chain(salt, s, _W - 1) for s in sk]
    return sk, pk, salt


def sign(sk: list[int], salt: int, digest: list[int]) -> list[int]:
    return [_chain(salt, s, d) for s, d in zip(sk, digest, strict=True)]


def verify(sig: list[int], salt: int, digest: list[int], pk: list[int]) -> bool:
    return all(_chain(salt, si, _W - 1 - d) == p for si, d, p in zip(sig, digest, pk, strict=True))


def _digest(rng: random.Random) -> list[int]:
    return [rng.randrange(_W) for _ in range(_N)]


def bench_winternitz_ots(seed: int = 20261231 + 421) -> dict[str, float]:
    rng = random.Random(seed)
    ok = forge = reuse = 0
    trials = 30
    for _ in range(trials):
        sk, pk, salt = keygen(rng)
        dig = _digest(rng)
        sig = sign(sk, salt, dig)
        ok += int(verify(sig, salt, dig, pk))
        # W-OTS security property: chains are one-way DOWNWARD — given
        # sig element chain(sk, d) you cannot produce chain(sk, d-1).
        # Pick a nibble with d>0, forge attempt: present existing chain
        # element for a smaller digest → rejected (needs preimage).
        idxs = [i for i, d in enumerate(dig) if d > 0]
        if idxs:
            i = rng.choice(idxs)
            sig2 = list(sig)
            sig2[i] = _h(salt, sig2[i])  # wrong element either way
            dig2 = list(dig)
            dig2[i] = dig[i] - 1
            forge += int(not verify(sig2, salt, dig2, pk))
        else:
            forge += 1
        # honest verifier rejects mismatched digest
        dig3 = list(dig)
        j = rng.randrange(_N)
        dig3[j] = (dig3[j] + 1) % _W
        reuse += int(not verify(sig, salt, dig3, pk))
    return {
        "synthetic_verifies": float(ok / trials),
        "synthetic_forgery_rejected": float(forge / trials),
        "synthetic_wrong_msg_rejected": float(reuse / trials),
    }
