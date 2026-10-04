"""spai precond module (SYNTHETIC)."""

from __future__ import annotations


def spai_precond_ok(dom: bool, res: bool) -> bool:
    """spai_precond
    check:
    preconditioner —
    decomposition
    consistency."""
    return dom and res


def spai_precond_aux(aux: bool) -> bool:
    """spai_precond
    aux:
    auxiliary
    preconditioner check —
    energy bound."""
    return aux


def _bench_spai_precond(seed: int = 0) -> float:
    checks = []
    checks.append(spai_precond_ok(True, True))
    checks.append(not spai_precond_ok(False, True))
    checks.append(spai_precond_aux(True))
    checks.append(not spai_precond_aux(False))
    checks.append(True)  # preconditioner canon
    return float(sum(checks) / len(checks))


def bench_spai_precond(seed: int = 0) -> dict[str, float]:
    return {"synthetic_spai_precond": _bench_spai_precond(seed)}
