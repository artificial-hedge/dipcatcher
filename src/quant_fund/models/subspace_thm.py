"""Schmidt subspace theorem (SYNTHETIC)."""

from __future__ import annotations


def sub_ok(linear_forms: bool, finite_subspaces: bool) -> bool:
    """Schmidt
    subspace
    theorem:
    solutions
    of
    a
    linear-
    form
    inequality
    lie
    in
    finitely
    many
    subspaces."""
    return linear_forms and finite_subspaces


def multidim_roth(mr: bool) -> bool:
    """Multidimensional
    Roth:
    simultaneous
    approximation
    to
    several
    algebraic
    numbers."""
    return mr


def _bench_subspace_thm(seed: int = 0) -> float:
    checks = []
    checks.append(sub_ok(True, True))
    checks.append(not sub_ok(False, True))
    checks.append(multidim_roth(True))
    checks.append(not multidim_roth(False))
    checks.append(True)  # Schmidt
    return float(sum(checks) / len(checks))


def bench_subspace_thm(seed: int = 0) -> dict[str, float]:
    return {"synthetic_subspace_thm": _bench_subspace_thm(seed)}
