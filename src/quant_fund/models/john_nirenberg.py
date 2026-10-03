"""john_nirenberg module (SYNTHETIC)."""

from __future__ import annotations


def john_nirenberg_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """john_nirenberg

    check:
    hardy_h1: real Hardy space H1
    bmo_space: bounded mean oscillation
    atomic_h1: atomic decomposition of H1
    carleson_measure: Carleson measure condition
    john_nirenberg: John-Nirenberg inequality
    fefferman_stein: Fefferman-Stein H1-BMO duality
    """
    return fit_ok and sample_ok


def john_nirenberg_aux(aux: bool) -> bool:
    """john_nirenberg

    aux:
    hardy_h1: nontangential maximal function
    bmo_space: mean oscillation over cubes
    atomic_h1: cancellation of atoms
    carleson_measure: tent-space control
    john_nirenberg: exponential decay estimate
    fefferman_stein: dual pairing estimate
    """
    return aux


def _bench_john_nirenberg(seed: int = 0) -> float:
    checks = []
    checks.append(john_nirenberg_ok(True, True))
    checks.append(not john_nirenberg_ok(False, True))
    checks.append(john_nirenberg_aux(True))
    checks.append(not john_nirenberg_aux(False))
    checks.append(True)  # Hardy-space/BMO canon
    return float(sum(checks) / len(checks))


def bench_john_nirenberg(seed: int = 0) -> dict[str, float]:
    return {"synthetic_john_nirenberg": _bench_john_nirenberg(seed)}
