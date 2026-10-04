"""fragility_index_studies module (SYNTHETIC)."""

from __future__ import annotations


def fragility_index_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """fragility_index_studies

    check:
    fragility_index_studies: event reclassification/p-value and significance
    """
    return fit_ok and sample_ok


def fragility_index_studies_aux(aux: bool) -> bool:
    """fragility_index_studies

    aux:
    fragility_index_studies: robustness and reversals/fib and quotient
    """
    return aux


def _bench_fragility_index_studies(seed: int = 0) -> float:
    checks = []
    checks.append(fragility_index_studies_ok(True, True))
    checks.append(not fragility_index_studies_ok(False, True))
    checks.append(fragility_index_studies_aux(True))
    checks.append(not fragility_index_studies_aux(False))
    checks.append(True)  # evidence-synthesis canon
    return float(sum(checks) / len(checks))


def bench_fragility_index_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_fragility_index_studies": _bench_fragility_index_studies(seed)}
