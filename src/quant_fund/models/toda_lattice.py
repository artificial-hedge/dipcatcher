"""toda_lattice module (SYNTHETIC)."""

from __future__ import annotations


def toda_lattice_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """toda_lattice

    check:
    sine_gordon: sine-Gordon equation
    nls_soliton: nonlinear Schroedinger solitons
    toda_lattice: Toda lattice
    calogero_moser: Calogero-Moser systems
    kp_hierarchy: Kadomtsev-Petviashvili hierarchy
    painleve_eq: Painleve equations
    """
    return fit_ok and sample_ok


def toda_lattice_aux(aux: bool) -> bool:
    """toda_lattice

    aux:
    sine_gordon: kink solutions
    nls_soliton: bright/dark solitons
    toda_lattice: Flaschka variables
    calogero_moser: root-system integrability
    kp_hierarchy: tau functions
    painleve_eq: Painleve property
    """
    return aux


def _bench_toda_lattice(seed: int = 0) -> float:
    checks = []
    checks.append(toda_lattice_ok(True, True))
    checks.append(not toda_lattice_ok(False, True))
    checks.append(toda_lattice_aux(True))
    checks.append(not toda_lattice_aux(False))
    checks.append(True)  # integrable-systems canon
    return float(sum(checks) / len(checks))


def bench_toda_lattice(seed: int = 0) -> dict[str, float]:
    return {"synthetic_toda_lattice": _bench_toda_lattice(seed)}
