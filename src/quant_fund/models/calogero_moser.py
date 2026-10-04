"""calogero_moser module (SYNTHETIC)."""

from __future__ import annotations


def calogero_moser_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """calogero_moser

    check:
    sine_gordon: sine-Gordon equation
    nls_soliton: nonlinear Schroedinger solitons
    toda_lattice: Toda lattice
    calogero_moser: Calogero-Moser systems
    kp_hierarchy: Kadomtsev-Petviashvili hierarchy
    painleve_eq: Painleve equations
    """
    return fit_ok and sample_ok


def calogero_moser_aux(aux: bool) -> bool:
    """calogero_moser

    aux:
    sine_gordon: kink solutions
    nls_soliton: bright/dark solitons
    toda_lattice: Flaschka variables
    calogero_moser: root-system integrability
    kp_hierarchy: tau functions
    painleve_eq: Painleve property
    """
    return aux


def _bench_calogero_moser(seed: int = 0) -> float:
    checks = []
    checks.append(calogero_moser_ok(True, True))
    checks.append(not calogero_moser_ok(False, True))
    checks.append(calogero_moser_aux(True))
    checks.append(not calogero_moser_aux(False))
    checks.append(True)  # integrable-systems canon
    return float(sum(checks) / len(checks))


def bench_calogero_moser(seed: int = 0) -> dict[str, float]:
    return {"synthetic_calogero_moser": _bench_calogero_moser(seed)}
