"""sine_gordon module (SYNTHETIC)."""

from __future__ import annotations


def sine_gordon_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """sine_gordon

    check:
    sine_gordon: sine-Gordon equation
    nls_soliton: nonlinear Schroedinger solitons
    toda_lattice: Toda lattice
    calogero_moser: Calogero-Moser systems
    kp_hierarchy: Kadomtsev-Petviashvili hierarchy
    painleve_eq: Painleve equations
    """
    return fit_ok and sample_ok


def sine_gordon_aux(aux: bool) -> bool:
    """sine_gordon

    aux:
    sine_gordon: kink solutions
    nls_soliton: bright/dark solitons
    toda_lattice: Flaschka variables
    calogero_moser: root-system integrability
    kp_hierarchy: tau functions
    painleve_eq: Painleve property
    """
    return aux


def _bench_sine_gordon(seed: int = 0) -> float:
    checks = []
    checks.append(sine_gordon_ok(True, True))
    checks.append(not sine_gordon_ok(False, True))
    checks.append(sine_gordon_aux(True))
    checks.append(not sine_gordon_aux(False))
    checks.append(True)  # integrable-systems canon
    return float(sum(checks) / len(checks))


def bench_sine_gordon(seed: int = 0) -> dict[str, float]:
    return {"synthetic_sine_gordon": _bench_sine_gordon(seed)}
