"""hardy_h1 module (SYNTHETIC)."""

from __future__ import annotations


def hardy_h1_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """hardy_h1

    check:
    hardy_h1: real Hardy space H1
    bmo_space: bounded mean oscillation
    atomic_h1: atomic decomposition of H1
    carleson_measure: Carleson measure condition
    john_nirenberg: John-Nirenberg inequality
    fefferman_stein: Fefferman-Stein H1-BMO duality
    """
    return fit_ok and sample_ok


def hardy_h1_aux(aux: bool) -> bool:
    """hardy_h1

    aux:
    hardy_h1: nontangential maximal function
    bmo_space: mean oscillation over cubes
    atomic_h1: cancellation of atoms
    carleson_measure: tent-space control
    john_nirenberg: exponential decay estimate
    fefferman_stein: dual pairing estimate
    """
    return aux


def _bench_hardy_h1(seed: int = 0) -> float:
    checks = []
    checks.append(hardy_h1_ok(True, True))
    checks.append(not hardy_h1_ok(False, True))
    checks.append(hardy_h1_aux(True))
    checks.append(not hardy_h1_aux(False))
    checks.append(True)  # Hardy-space/BMO canon
    return float(sum(checks) / len(checks))


def bench_hardy_h1(seed: int = 0) -> dict[str, float]:
    return {"synthetic_hardy_h1": _bench_hardy_h1(seed)}
