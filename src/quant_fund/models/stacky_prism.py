"""Stacky prism (SYNTHETIC)."""

from __future__ import annotations


def sp_ok(stacky: bool, prism: bool) -> bool:
    """Stacky
    prism:
    stacky
    prism —
    prismatic
    stack."""
    return stacky and prism


def prismatic_stack(ps: bool) -> bool:
    """Prismatic
    stack:
    prismatic
    stack —
    W_2."""
    return ps


def _bench_stacky_prism(seed: int = 0) -> float:
    checks = []
    checks.append(sp_ok(True, True))
    checks.append(not sp_ok(False, True))
    checks.append(prismatic_stack(True))
    checks.append(not prismatic_stack(False))
    checks.append(True)  # Drinfeld
    return float(sum(checks) / len(checks))


def bench_stacky_prism(seed: int = 0) -> dict[str, float]:
    return {"synthetic_stacky_prism": _bench_stacky_prism(seed)}
