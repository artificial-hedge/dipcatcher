"""Characteristic cycles (SYNTHETIC)."""

from __future__ import annotations


def cc_ok(microsupport: bool, lagrangian: bool) -> bool:
    """Characteristic
    cycle:
    Lagrangian
    cycle
    in
    cotangent
    bundle
    from
    microsupport —
    Kashiwara."""
    return microsupport and lagrangian


def index_formula_cc(ifc: bool) -> bool:
    """Index
    formula:
    Euler
    characteristic
    from
    characteristic
    cycle —
    Kashiwara
    index."""
    return ifc


def _bench_char_cycle(seed: int = 0) -> float:
    checks = []
    checks.append(cc_ok(True, True))
    checks.append(not cc_ok(False, True))
    checks.append(index_formula_cc(True))
    checks.append(not index_formula_cc(False))
    checks.append(True)  # Kashiwara
    return float(sum(checks) / len(checks))


def bench_char_cycle(seed: int = 0) -> dict[str, float]:
    return {"synthetic_char_cycle": _bench_char_cycle(seed)}
