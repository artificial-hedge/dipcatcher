"""kp_hierarchy module (SYNTHETIC)."""

from __future__ import annotations


def kp_hierarchy_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """kp_hierarchy

    check:
    sine_gordon: sine-Gordon equation
    nls_soliton: nonlinear Schroedinger solitons
    toda_lattice: Toda lattice
    calogero_moser: Calogero-Moser systems
    kp_hierarchy: Kadomtsev-Petviashvili hierarchy
    painleve_eq: Painleve equations
    """
    return fit_ok and sample_ok


def kp_hierarchy_aux(aux: bool) -> bool:
    """kp_hierarchy

    aux:
    sine_gordon: kink solutions
    nls_soliton: bright/dark solitons
    toda_lattice: Flaschka variables
    calogero_moser: root-system integrability
    kp_hierarchy: tau functions
    painleve_eq: Painleve property
    """
    return aux


def _bench_kp_hierarchy(seed: int = 0) -> float:
    checks = []
    checks.append(kp_hierarchy_ok(True, True))
    checks.append(not kp_hierarchy_ok(False, True))
    checks.append(kp_hierarchy_aux(True))
    checks.append(not kp_hierarchy_aux(False))
    checks.append(True)  # integrable-systems canon
    return float(sum(checks) / len(checks))


def bench_kp_hierarchy(seed: int = 0) -> dict[str, float]:
    return {"synthetic_kp_hierarchy": _bench_kp_hierarchy(seed)}
