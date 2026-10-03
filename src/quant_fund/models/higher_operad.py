"""Higher operads (SYNTHETIC)."""

from __future__ import annotations


def ho_ok(higher: bool, operad: bool) -> bool:
    """Higher
    operad:
    higher
    operad —
    higher
    categorical."""
    return higher and operad


def operadic_tree(ot: bool) -> bool:
    """Operadic
    tree:
    operadic
    tree
    grafting —
    tree
    operad."""
    return ot


def _bench_higher_operad(seed: int = 0) -> float:
    checks = []
    checks.append(ho_ok(True, True))
    checks.append(not ho_ok(False, True))
    checks.append(operadic_tree(True))
    checks.append(not operadic_tree(False))
    checks.append(True)  # May
    return float(sum(checks) / len(checks))


def bench_higher_operad(seed: int = 0) -> dict[str, float]:
    return {"synthetic_higher_operad": _bench_higher_operad(seed)}
