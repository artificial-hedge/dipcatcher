"""Toy ElGamal encryption over a safe-prime group; roundtrip + homomorphic check."""

import numpy as np

_SEED = 20261231 + 692


def _modexp(b: int, e: int, m: int) -> int:
    return pow(b, e, m)


_P = 104729  # prime
_G = 2


def keygen(rng: np.random.RandomState) -> tuple[int, int]:
    sk = int(rng.randint(2, _P - 2))
    return sk, _modexp(_G, sk, _P)


def enc(pk: int, m: int, r: int) -> tuple[int, int]:
    return _modexp(_G, r, _P), (m * _modexp(pk, r, _P)) % _P


def dec(sk: int, ct: tuple[int, int]) -> int:
    c1, c2 = ct
    s = _modexp(c1, sk, _P)
    return (c2 * pow(s, -1, _P)) % _P


def bench_elgamal_enc(seed: int = _SEED) -> dict[str, float]:
    rng = np.random.RandomState(seed)
    ok = 0.0
    trials = 30
    for _ in range(trials):
        sk, pk = keygen(rng)
        m1 = int(rng.randint(2, 200))
        m2 = int(rng.randint(2, 200))
        c1 = enc(pk, m1, int(rng.randint(2, _P - 2)))
        c2 = enc(pk, m2, int(rng.randint(2, _P - 2)))
        ok += float(dec(sk, c1) == m1 and dec(sk, c2) == m2)
        # multiplicative homomorphism: dec(c1*c2) == m1*m2 mod p
        prod = (c1[0] * c2[0] % _P, c1[1] * c2[1] % _P)
        ok += float(dec(sk, prod) == (m1 * m2) % _P)
    return {"synthetic_elgamal_roundtrip": ok / (2 * trials)}
