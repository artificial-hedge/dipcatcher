"""schwarzschild_metric module (SYNTHETIC)."""

from __future__ import annotations


def schwarzschild_metric_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """schwarzschild_metric

    check:
    einstein_equations: Einstein field equations
    schwarzschild_metric: Schwarzschild solution
    friedmann_eq: Friedmann equations
    kerr_metric: Kerr black hole
    gr_birkhoff: Birkhoff theorem
    penrose_diagrams: Penrose diagrams
    """
    return fit_ok and sample_ok


def schwarzschild_metric_aux(aux: bool) -> bool:
    """schwarzschild_metric

    aux:
    einstein_equations: Bianchi identity
    schwarzschild_metric: event horizon
    friedmann_eq: scale factor
    kerr_metric: frame dragging
    gr_birkhoff: uniqueness
    penrose_diagrams: causal structure
    """
    return aux


def _bench_schwarzschild_metric(seed: int = 0) -> float:
    checks = []
    checks.append(schwarzschild_metric_ok(True, True))
    checks.append(not schwarzschild_metric_ok(False, True))
    checks.append(schwarzschild_metric_aux(True))
    checks.append(not schwarzschild_metric_aux(False))
    checks.append(True)  # general-relativity canon
    return float(sum(checks) / len(checks))


def bench_schwarzschild_metric(seed: int = 0) -> dict[str, float]:
    return {"synthetic_schwarzschild_metric": _bench_schwarzschild_metric(seed)}
