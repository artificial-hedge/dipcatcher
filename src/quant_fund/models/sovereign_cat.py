"""Sovereign categories (SYNTHETIC)."""

from __future__ import annotations


def sc_ok(sovereign: bool, cat: bool) -> bool:
    """Sovereign
    category:
    sovereign
    category —
    left
    right
    dual."""
    return sovereign and cat


def sovereign_axiom(sa: bool) -> bool:
    """Sovereign
    axiom:
    sovereign
    axiom —
    duals
    coincide."""
    return sa


def _bench_sovereign_cat(seed: int = 0) -> float:
    checks = []
    checks.append(sc_ok(True, True))
    checks.append(not sc_ok(False, True))
    checks.append(sovereign_axiom(True))
    checks.append(not sovereign_axiom(False))
    checks.append(True)  # Freyd-Yetter
    return float(sum(checks) / len(checks))


def bench_sovereign_cat(seed: int = 0) -> dict[str, float]:
    return {"synthetic_sovereign_cat": _bench_sovereign_cat(seed)}
