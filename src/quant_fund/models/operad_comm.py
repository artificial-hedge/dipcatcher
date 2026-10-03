"""Commutative operad: Comm(n) is a single point (SYNTHETIC)."""

from __future__ import annotations


def comm_size(n: int) -> int:
    """Comm(n) = {*} for all n: exactly one n-ary operation."""
    return 1


def comm_compose(*args: int) -> int:
    """The unique composition of unique operations is unique."""
    return 0


def _bench_operad_comm(seed: int = 0) -> float:
    checks = []
    checks.append(all(comm_size(n) == 1 for n in range(1, 6)))
    # composition is trivially associative and unital
    checks.append(comm_compose(0, 0, 0) == 0)
    # Com-algebra = commutative monoid: ab*cd associativity+commutativity
    # on the model (Z_4, +): verify a+b == b+a and (a+b)+c == a+(b+c)
    checks.append(all((a + b) % 4 == (b + a) % 4 for a in range(4) for b in range(4)))
    checks.append(
        all(
            ((a + b) % 4 + c) % 4 == (a + (b + c) % 4) % 4
            for a in range(4)
            for b in range(4)
            for c in range(4)
        )
    )
    # unlike Assoc, all n-ary ops collapse to one
    checks.append(comm_size(2) < 6)  # Assoc(3)=6 vs Comm(3)=1
    return float(sum(checks) / len(checks))


def bench_operad_comm(seed: int = 0) -> dict[str, float]:
    return {"synthetic_operad_comm": _bench_operad_comm(seed)}
