"""Toy Paillier additive homomorphic encryption; roundtrip + E(m1+m2) check (SYNTHETIC)."""

import math

import numpy as np

_SEED = 20261231 + 693

_P, _Q = 104729, 104759  # primes
_N = _P * _Q
_NN = _N * _N
_G = _N + 1


def _lam(p: int, q: int) -> int:
    return (p - 1) * (q - 1) // math.gcd(p - 1, q - 1)


def paillier_enc(m: int, r: int) -> int:
    return (pow(_G, m, _NN) * pow(r, _N, _NN)) % _NN


def paillier_dec(c: int, lam: int) -> int:
    u = pow(c, lam, _NN)
    l_val = (u - 1) // _N
    return (l_val * pow((pow(_G, lam, _NN) - 1) // _N, -1, _N)) % _N


def bench_paillier_he(seed: int = _SEED) -> dict[str, float]:
    rng = np.random.RandomState(seed)
    lam = _lam(_P, _Q)
    ok = 0.0
    trials = 15
    for _ in range(trials):
        m1 = int(rng.randint(1, 10_000))
        m2 = int(rng.randint(1, 10_000))
        r1 = int(rng.randint(1, _N - 1))
        r2 = int(rng.randint(1, _N - 1))
        c1 = paillier_enc(m1, r1)
        c2 = paillier_enc(m2, r2)
        ok += float(paillier_dec(c1, lam) == m1 % _N)
        ok += float(paillier_dec(c1 * c2 % _NN, lam) == (m1 + m2) % _N)
    return {"synthetic_paillier_roundtrip": ok / (2 * trials)}
