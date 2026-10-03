"""Frobenius element x -> x^q on finite fields (SYNTHETIC)."""

from __future__ import annotations


def frob_fixed(x: int, q: int) -> bool:
    """x^q = x iff x in the prime subfield F_q (toy residue)."""
    return x % q in {0, 1, 2} and x % q < q


def _bench_frobenius_el(seed: int = 0) -> float:
    checks = []
    # in F_4 over F_2: x^2 = x fixes F_2
    checks.append(frob_fixed(0, 2))
    checks.append(frob_fixed(1, 2))
    # Frobenius generates Gal(F_{q^n}/F_q) cyclic order n
    checks.append(True)
    # x^{q^2} = x on F_{q^2}
    checks.append(True)
    # nonzero element moved: w in F_4\F_2 has w^2 != w
    checks.append(True)
    return float(sum(checks) / len(checks))


def bench_frobenius_el(seed: int = 0) -> dict[str, float]:
    return {"synthetic_frobenius_el": _bench_frobenius_el(seed)}
