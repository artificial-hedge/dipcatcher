"""Shamir (k, n) secret sharing over GF(p).

Polynomial f(x) = s + a1 x + ... + a_{k-1} x^{k-1} mod p; any k shares
reconstruct s via Lagrange interpolation at x = 0. Bench: reconstruction
exactness from every k-subset of shares, and failure (wrong secret) when
only k-1 shares are pooled.
"""

from itertools import combinations

import numpy as np

_P = 99991  # prime field for sharing


def _share(seed: int, secret: int, k: int, n: int) -> list[tuple[int, int]]:
    rng = np.random.default_rng(seed)
    coef = [secret] + [int(x) for x in rng.integers(0, _P, k - 1)]
    return [(x, int(np.polyval(coef[::-1], x) % _P)) for x in range(1, n + 1)]


def _recon(shares: list[tuple[int, int]]) -> int:
    s = 0
    for i, (xi, yi) in enumerate(shares):
        num, den = 1, 1
        for j, (xj, _) in enumerate(shares):
            if i == j:
                continue
            num = num * (-xj) % _P
            den = den * (xi - xj) % _P
        s = (s + yi * num * pow(den, _P - 2, _P)) % _P
    return s


def bench_shamir_secret(seed: int = 4705) -> dict[str, float]:
    secret = 42424
    k, n = 3, 5
    shares = _share(seed, secret, k, n)
    oks = 0
    for combo in combinations(shares, k):
        oks += int(_recon(list(combo)) == secret)
    # k-1 shares should NOT reconstruct
    bad = _recon(shares[: k - 1])
    return {
        "synthetic_shamir_exact": float(oks),
        "synthetic_shamir_combos": float(len(list(combinations(shares, k)))),
        "synthetic_shamir_under_err": float(bad != secret),
        "synthetic_shamir_wrong_val": float(bad),
        "synthetic_shamir_p": float(_P),
    }
