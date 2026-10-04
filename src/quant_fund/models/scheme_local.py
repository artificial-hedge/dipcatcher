"""Spec and local rings of finite rings (SYNTHETIC)."""

from __future__ import annotations


def prime_ideals_zn(n: int) -> list[int]:
    """Prime ideals of Z/nZ are (p) for primes p | n (maximal = prime here)."""
    out = []
    d = 2
    m = n
    while d * d <= m:
        if m % d == 0:
            out.append(d)
            while m % d == 0:
                m //= d
        d += 1
    if m > 1:
        out.append(m)
    return out


def local_ring_size(n: int, p: int) -> int:
    """Localization of Z/nZ at the prime (p) is Z/p^{v_p(n)}Z — the p-primary
    component of n."""
    v = 0
    m = n
    while m % p == 0:
        m //= p
        v += 1
    return int(p**v)


def _bench_scheme_local(seed: int = 0) -> float:
    checks = []
    # Spec Z/12 = {(2), (3)}: two points
    checks.append(prime_ideals_zn(12) == [2, 3])
    # Spec Z/30 = {2,3,5}
    checks.append(prime_ideals_zn(30) == [2, 3, 5])
    # Spec Z/7 = single point (field)
    checks.append(prime_ideals_zn(7) == [7])
    # local ring of Z/12 at (2) = Z/4, at (3) = Z/3
    checks.append(local_ring_size(12, 2) == 4)
    checks.append(local_ring_size(12, 3) == 3)
    # product of local rings = Z/12: 4*3 = 12 (CRT)
    checks.append(local_ring_size(12, 2) * local_ring_size(12, 3) == 12)
    # Z/36 at (3) = Z/9, at (2) = Z/4; 9*4 = 36
    checks.append(local_ring_size(36, 3) == 9 and local_ring_size(36, 2) == 4)
    return float(sum(checks) / len(checks))


def bench_scheme_local(seed: int = 0) -> dict[str, float]:
    return {"synthetic_scheme_local": _bench_scheme_local(seed)}
