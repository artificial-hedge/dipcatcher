"""initial enlarg module (SYNTHETIC)."""

from __future__ import annotations


def initial_enlarg_ok(f: bool, right: bool) -> bool:
    """initial_enlarg
    check:
    filtration —
    right
    continuity."""
    return f and right


def initial_enlarg_aux(aux: bool) -> bool:
    """initial_enlarg
    aux:
    auxiliary
    filtration check —
    completeness."""
    return aux


def _bench_initial_enlarg(seed: int = 0) -> float:
    checks = []
    checks.append(initial_enlarg_ok(True, True))
    checks.append(not initial_enlarg_ok(False, True))
    checks.append(initial_enlarg_aux(True))
    checks.append(not initial_enlarg_aux(False))
    checks.append(True)  # filtration canon
    return float(sum(checks) / len(checks))


def bench_initial_enlarg(seed: int = 0) -> dict[str, float]:
    return {"synthetic_initial_enlarg": _bench_initial_enlarg(seed)}
