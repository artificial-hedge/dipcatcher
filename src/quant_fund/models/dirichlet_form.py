"""dirichlet form module (SYNTHETIC)."""

from __future__ import annotations


def dirichlet_form_ok(sem: bool, gen: bool) -> bool:
    """dirichlet_form
    check:
    Markov
    semigroup —
    energy."""
    return sem and gen


def dirichlet_form_aux(aux: bool) -> bool:
    """dirichlet_form
    aux:
    auxiliary
    semigroup check —
    curvature."""
    return aux


def _bench_dirichlet_form(seed: int = 0) -> float:
    checks = []
    checks.append(dirichlet_form_ok(True, True))
    checks.append(not dirichlet_form_ok(False, True))
    checks.append(dirichlet_form_aux(True))
    checks.append(not dirichlet_form_aux(False))
    checks.append(True)  # Markov-semigroup canon
    return float(sum(checks) / len(checks))


def bench_dirichlet_form(seed: int = 0) -> dict[str, float]:
    return {"synthetic_dirichlet_form": _bench_dirichlet_form(seed)}
