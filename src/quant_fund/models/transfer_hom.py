"""Transfer homomorphism (Verlagerung) bookkeeping (SYNTHIC) (SYNTHETIC)."""

from __future__ import annotations

from math import gcd


def verlagerung_order(x: int, g_order: int, h_order: int, h_abel: bool) -> int:
    """The transfer G -> H/[H,H] sends x to x^{[G:H]} when H is central.
    Order of the image of x divides h_order; here we model the exponent."""
    idx = g_order // h_order
    return pow(x, idx, max(h_order, 1)) if h_order else 0


def transfer_cyclic_subgroup(g_order: int, h_order: int) -> int:
    """For a cyclic group C_g and subgroup C_h, Ver(g) = g^{g/h}: the
    exponent applied to a generator."""
    return g_order // h_order


def _bench_transfer_hom(seed: int = 0) -> float:
    checks = []
    # C3 <= S3 is not central; model the central case C3 <= C6:
    # transfer of generator x of C6 to C3 = x^{[C6:C3]} = x^2
    checks.append(transfer_cyclic_subgroup(6, 3) == 2)
    # C2 <= C4: exponent 2 -> Ver(g) = g^2 (the element of order 2)
    checks.append(transfer_cyclic_subgroup(4, 2) == 2)
    # C5 <= C15: exponent 3
    checks.append(transfer_cyclic_subgroup(15, 5) == 3)
    # verlagerung_order: x=1 (generator residue) into C3, [G:H]=2 -> 1^2=1
    checks.append(verlagerung_order(1, 6, 3, True) == 1)
    # gcd sanity on exponents for p-subgroups: exponent divides index
    checks.append(gcd(transfer_cyclic_subgroup(9, 3), 3) == 3)
    return float(sum(checks) / len(checks))


def bench_transfer_hom(seed: int = 0) -> dict[str, float]:
    return {"synthetic_transfer_hom": _bench_transfer_hom(seed)}
