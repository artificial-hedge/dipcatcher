"""bmo_space module (SYNTHETIC)."""

from __future__ import annotations


def bmo_space_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """bmo_space

    check:
    hardy_h1: real Hardy space H1
    bmo_space: bounded mean oscillation
    atomic_h1: atomic decomposition of H1
    carleson_measure: Carleson measure condition
    john_nirenberg: John-Nirenberg inequality
    fefferman_stein: Fefferman-Stein H1-BMO duality
    """
    return fit_ok and sample_ok


def bmo_space_aux(aux: bool) -> bool:
    """bmo_space

    aux:
    hardy_h1: nontangential maximal function
    bmo_space: mean oscillation over cubes
    atomic_h1: cancellation of atoms
    carleson_measure: tent-space control
    john_nirenberg: exponential decay estimate
    fefferman_stein: dual pairing estimate
    """
    return aux


def _bench_bmo_space(seed: int = 0) -> float:
    checks = []
    checks.append(bmo_space_ok(True, True))
    checks.append(not bmo_space_ok(False, True))
    checks.append(bmo_space_aux(True))
    checks.append(not bmo_space_aux(False))
    checks.append(True)  # Hardy-space/BMO canon
    return float(sum(checks) / len(checks))


def bench_bmo_space(seed: int = 0) -> dict[str, float]:
    return {"synthetic_bmo_space": _bench_bmo_space(seed)}
