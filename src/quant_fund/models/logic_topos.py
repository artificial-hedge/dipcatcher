"""Internal logic of Set: Boolean truth values (SYNTHETIC)."""

from __future__ import annotations


def internal_and(a: bool, b: bool) -> bool:
    return a and b


def internal_implies(a: bool, b: bool) -> bool:
    return (not a) or b


def _bench_logic_topos(seed: int = 0) -> float:
    checks = []
    # Omega in Set = {true, false}: Boolean
    checks.append(internal_and(True, False) is False)
    checks.append(internal_implies(True, False) is False)
    checks.append(internal_implies(False, False) is True)
    # excluded middle holds in Set: b v ~b is true for both values
    def excl_mid(b: bool) -> bool:
        return b if b else not b

    checks.append(excl_mid(True) and excl_mid(False))
    # subobject negation: chi of complement = not chi
    checks.append((not True) is False)
    return float(sum(checks) / len(checks))


def bench_logic_topos(seed: int = 0) -> dict[str, float]:
    return {"synthetic_logic_topos": _bench_logic_topos(seed)}
