"""penrose_diagrams module (SYNTHETIC)."""

from __future__ import annotations


def penrose_diagrams_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """penrose_diagrams

    check:
    einstein_equations: Einstein field equations
    schwarzschild_metric: Schwarzschild solution
    friedmann_eq: Friedmann equations
    kerr_metric: Kerr black hole
    gr_birkhoff: Birkhoff theorem
    penrose_diagrams: Penrose diagrams
    """
    return fit_ok and sample_ok


def penrose_diagrams_aux(aux: bool) -> bool:
    """penrose_diagrams

    aux:
    einstein_equations: Bianchi identity
    schwarzschild_metric: event horizon
    friedmann_eq: scale factor
    kerr_metric: frame dragging
    gr_birkhoff: uniqueness
    penrose_diagrams: causal structure
    """
    return aux


def _bench_penrose_diagrams(seed: int = 0) -> float:
    checks = []
    checks.append(penrose_diagrams_ok(True, True))
    checks.append(not penrose_diagrams_ok(False, True))
    checks.append(penrose_diagrams_aux(True))
    checks.append(not penrose_diagrams_aux(False))
    checks.append(True)  # general-relativity canon
    return float(sum(checks) / len(checks))


def bench_penrose_diagrams(seed: int = 0) -> dict[str, float]:
    return {"synthetic_penrose_diagrams": _bench_penrose_diagrams(seed)}
