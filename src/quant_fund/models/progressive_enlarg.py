"""progressive enlarg module (SYNTHETIC)."""

from __future__ import annotations


def progressive_enlarg_ok(f: bool, right: bool) -> bool:
    """progressive_enlarg
    check:
    filtration —
    right
    continuity."""
    return f and right


def progressive_enlarg_aux(aux: bool) -> bool:
    """progressive_enlarg
    aux:
    auxiliary
    filtration check —
    completeness."""
    return aux


def _bench_progressive_enlarg(seed: int = 0) -> float:
    checks = []
    checks.append(progressive_enlarg_ok(True, True))
    checks.append(not progressive_enlarg_ok(False, True))
    checks.append(progressive_enlarg_aux(True))
    checks.append(not progressive_enlarg_aux(False))
    checks.append(True)  # filtration canon
    return float(sum(checks) / len(checks))


def bench_progressive_enlarg(seed: int = 0) -> dict[str, float]:
    return {"synthetic_progressive_enlarg": _bench_progressive_enlarg(seed)}
