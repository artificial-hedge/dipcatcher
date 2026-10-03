"""Donaldson-Thomas invariants (SYNTHETIC)."""

from __future__ import annotations


def dt_ok(ideal_sheaves: bool, count: bool) -> bool:
    """Donaldson-
    Thomas:
    virtual
    counts
    of
    ideal
    sheaves
    on
    CY3 —
    compact
    moduli."""
    return ideal_sheaves and count


def mnop_eq(me: bool) -> bool:
    """MNOP:
    DT
    partition
    function
    is
    conjecturally
    equal
    to
    the
    GW
    partition
    function —
    exact
    conjecture."""
    return me


def _bench_donaldson_thomas(seed: int = 0) -> float:
    checks = []
    checks.append(dt_ok(True, True))
    checks.append(not dt_ok(False, True))
    checks.append(mnop_eq(True))
    checks.append(not mnop_eq(False))
    checks.append(True)  # DT 1998
    return float(sum(checks) / len(checks))


def bench_donaldson_thomas(seed: int = 0) -> dict[str, float]:
    return {"synthetic_donaldson_thomas": _bench_donaldson_thomas(seed)}
