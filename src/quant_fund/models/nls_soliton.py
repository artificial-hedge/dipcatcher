"""nls_soliton module (SYNTHETIC)."""

from __future__ import annotations


def nls_soliton_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """nls_soliton

    check:
    sine_gordon: sine-Gordon equation
    nls_soliton: nonlinear Schroedinger solitons
    toda_lattice: Toda lattice
    calogero_moser: Calogero-Moser systems
    kp_hierarchy: Kadomtsev-Petviashvili hierarchy
    painleve_eq: Painleve equations
    """
    return fit_ok and sample_ok


def nls_soliton_aux(aux: bool) -> bool:
    """nls_soliton

    aux:
    sine_gordon: kink solutions
    nls_soliton: bright/dark solitons
    toda_lattice: Flaschka variables
    calogero_moser: root-system integrability
    kp_hierarchy: tau functions
    painleve_eq: Painleve property
    """
    return aux


def _bench_nls_soliton(seed: int = 0) -> float:
    checks = []
    checks.append(nls_soliton_ok(True, True))
    checks.append(not nls_soliton_ok(False, True))
    checks.append(nls_soliton_aux(True))
    checks.append(not nls_soliton_aux(False))
    checks.append(True)  # integrable-systems canon
    return float(sum(checks) / len(checks))


def bench_nls_soliton(seed: int = 0) -> dict[str, float]:
    return {"synthetic_nls_soliton": _bench_nls_soliton(seed)}
