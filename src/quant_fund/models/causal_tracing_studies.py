"""causal_tracing_studies module (SYNTHETIC)."""

from __future__ import annotations


def causal_tracing_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """causal_tracing_studies

    check:
    causal_tracing_studies: activation swaps and corrupted runs/states and restoration
    """
    return fit_ok and sample_ok


def causal_tracing_studies_aux(aux: bool) -> bool:
    """causal_tracing_studies

    aux:
    causal_tracing_studies: window and MLP effects/sites and contributions
    """
    return aux


def _bench_causal_tracing_studies(seed: int = 0) -> float:
    checks = []
    checks.append(causal_tracing_studies_ok(True, True))
    checks.append(not causal_tracing_studies_ok(False, True))
    checks.append(causal_tracing_studies_aux(True))
    checks.append(not causal_tracing_studies_aux(False))
    checks.append(True)  # mech-interp-2 canon
    return float(sum(checks) / len(checks))


def bench_causal_tracing_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_causal_tracing_studies": _bench_causal_tracing_studies(seed)}
