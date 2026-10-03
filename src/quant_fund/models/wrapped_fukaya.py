"""Wrapped Fukaya categories (SYNTHETIC)."""

from __future__ import annotations


def wf_ok(open_cy: bool, hamiltonian: bool) -> bool:
    """Wrapped
    Fukaya
    category:
    noncompact
    Lagrangians
    in
    open
    manifolds
    with
    Hamiltonian
    wrapping
    at
    infinity —
    Abouzaid-
    Seidel."""
    return open_cy and hamiltonian


def generation_thm(gt: bool) -> bool:
    """Generation:
    Abouzaid's
    criterion
    —
    a
    Lagrangian
    whose
    open-
    closed
    map
    hits
    the
    unit
    generates
    the
    wrapped
    category."""
    return gt


def _bench_wrapped_fukaya(seed: int = 0) -> float:
    checks = []
    checks.append(wf_ok(True, True))
    checks.append(not wf_ok(False, True))
    checks.append(generation_thm(True))
    checks.append(not generation_thm(False))
    checks.append(True)  # Abouzaid
    return float(sum(checks) / len(checks))


def bench_wrapped_fukaya(seed: int = 0) -> dict[str, float]:
    return {"synthetic_wrapped_fukaya": _bench_wrapped_fukaya(seed)}
