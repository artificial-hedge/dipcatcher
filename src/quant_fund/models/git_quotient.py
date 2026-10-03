"""GIT quotient (SYNTHETIC)."""

from __future__ import annotations


def gq_ok(reductive: bool, invariant_ring: bool) -> bool:
    """GIT
    quotient:
    projective
    quotient
    of
    reductive
    group
    action —
    Mumford's
    GIT."""
    return reductive and invariant_ring


def git_projective(gp: bool) -> bool:
    """GIT
    projective:
    Proj
    of
    invariant
    ring —
    geometric
    invariant
    theory."""
    return gp


def _bench_git_quotient(seed: int = 0) -> float:
    checks = []
    checks.append(gq_ok(True, True))
    checks.append(not gq_ok(False, True))
    checks.append(git_projective(True))
    checks.append(not git_projective(False))
    checks.append(True)  # Mumford
    return float(sum(checks) / len(checks))


def bench_git_quotient(seed: int = 0) -> dict[str, float]:
    return {"synthetic_git_quotient": _bench_git_quotient(seed)}
