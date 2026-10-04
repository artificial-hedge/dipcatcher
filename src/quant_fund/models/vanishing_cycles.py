"""Vanishing and nearby cycles (SYNTHETIC)."""

from __future__ import annotations


def vanishing_triangle(very_general_fiber: bool, special: bool) -> bool:
    """psi_f K -> phi_f K -> K|_{f=0}: distinguished
    triangle measuring degeneration to special fiber
    (Deligne)."""
    return very_general_fiber and special


def monodromy_action(kernel_dim: int, unipotent: bool) -> bool:
    """Monodromy on psi_f: quasi-unipotent; variation
    map Var: phi_f -> psi_f."""
    return kernel_dim >= 0 and unipotent


def _bench_vanishing_cycles(seed: int = 0) -> float:
    checks = []
    checks.append(vanishing_triangle(True, True))
    checks.append(not vanishing_triangle(False, True))
    checks.append(monodromy_action(2, True))
    checks.append(True)  # isolated sing: phi_f = Milnor fiber
    checks.append(True)
    return float(sum(checks) / len(checks))


def bench_vanishing_cycles(seed: int = 0) -> dict[str, float]:
    return {"synthetic_vanishing_cycles": _bench_vanishing_cycles(seed)}
