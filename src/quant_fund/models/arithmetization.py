"""Godel beta function: sequence coding (SYNTHETIC)."""

from __future__ import annotations


def beta(a: int, b: int, i: int) -> int:
    """beta(a, b, i) = a mod (1 + (i+1)*b) recovers the i-th element."""
    return a % (1 + (i + 1) * b)


def code_seq(seq: list[int]) -> tuple[int, int]:
    """Pick b > max(seq) factorial-ish and solve a by CRT over the
    pairwise-coprime moduli 1+(i+1)b for b = max+1! ... simple choice:
    b = m! where m > max elements ensures coprimality."""
    m = max(seq) + len(seq) + 1
    b = 1
    for k in range(2, m + 1):
        b *= k
    mods = [1 + (i + 1) * b for i in range(len(seq))]
    a = _crt(seq, mods)
    return (a, b)


def _crt(rems: list[int], mods: list[int]) -> int:
    x = 0
    prod = 1
    for m in mods:
        prod *= m
    for r, m in zip(rems, mods, strict=True):
        p = prod // m
        inv = pow(p, -1, m)
        x = (x + r * p * inv) % prod
    return x


def _bench_arithmetization(seed: int = 0) -> float:
    checks = []
    seq = [3, 1, 4, 1, 5]
    a, b = code_seq(seq)
    # every element is recovered by the beta function
    checks.append(all(beta(a, b, i) == seq[i] for i in range(len(seq))))
    # beta value bounded by modulus
    checks.append(beta(a, b, 0) < 1 + b)
    # different sequences code differently
    a2, b2 = code_seq([9, 9, 9])
    checks.append((a, b) != (a2, b2))
    # single element round-trips
    a3, b3 = code_seq([7])
    checks.append(beta(a3, b3, 0) == 7)
    # longer sequences also work
    seq2 = list(range(6))
    a4, b4 = code_seq(seq2)
    checks.append(all(beta(a4, b4, i) == i for i in range(6)))
    return float(sum(checks) / len(checks))


def bench_arithmetization(seed: int = 0) -> dict[str, float]:
    return {"synthetic_arithmetization": _bench_arithmetization(seed)}
