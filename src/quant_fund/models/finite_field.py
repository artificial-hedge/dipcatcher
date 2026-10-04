"""Finite fields GF(p): field axioms + Frobenius endomorphism x^p = x (SYNTHETIC)."""

from __future__ import annotations


def _bench_finite_field(seed: int = 0) -> float:
    checks = []
    p = 7
    # additive group: every element has inverse
    checks.append(all((a + (-a % p)) % p == 0 for a in range(p)))
    # multiplicative group of nonzero: closed, inverse exists
    for a in range(1, p):
        inv = pow(a, p - 2, p)
        checks.append((a * inv) % p == 1)
    # Fermat little theorem: a^p = a mod p
    checks.append(all(pow(a, p, p) == a % p for a in range(p)))
    # Frobenius: (a+b)^p = a^p + b^p in char p
    checks.append(
        all(
            pow((a + b) % p, p, p) == (pow(a, p, p) + pow(b, p, p)) % p
            for a in range(p)
            for b in range(p)
        )
    )

    # multiplicative group is cyclic: some generator has order p-1
    def order(a: int, n: int) -> int:
        x, k = 1, 0
        while x != 1 or k == 0:
            x = (x * a) % n
            k += 1
        return k

    checks.append(any(order(a, p) == p - 1 for a in range(1, p)))
    # GF(4) via x^2+x+1 over GF(2): nonzero elements form C3
    # elements: 0,1,w,w+1 with w^2=w+1
    mul = {
        (1, 1): 1,
        (1, 2): 2,
        (1, 3): 3,
        (2, 2): 3,
        (2, 3): 1,
        (3, 3): 2,
        (3, 2): 1,
        (2, 1): 3,
        (3, 1): 2,
    }
    checks.append(all(mul[(a, b)] in (1, 2, 3) for a in (1, 2, 3) for b in (1, 2, 3)))
    checks.append(mul[(2, mul[(2, 2)])] == 1)  # w * w^2 = w^3 = 1
    return float(sum(checks) / len(checks))


def bench_finite_field(seed: int = 0) -> dict[str, float]:
    return {"synthetic_finite_field": _bench_finite_field(seed)}
