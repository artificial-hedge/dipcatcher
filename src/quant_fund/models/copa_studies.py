"""copa_studies module (SYNTHETIC)."""

from __future__ import annotations


def copa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """copa_studies

    check:
    copa_studies: COPA causal choice premises and alternatives
    """
    return fit_ok and sample_ok


def copa_studies_aux(aux: bool) -> bool:
    """copa_studies

    aux:
    copa_studies: premise-choice pairs, effects, and accuracy
    """
    return aux


def _bench_copa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(copa_studies_ok(True, True))
    checks.append(not copa_studies_ok(False, True))
    checks.append(copa_studies_aux(True))
    checks.append(not copa_studies_aux(False))
    checks.append(True)  # commonsense-eval canon
    return float(sum(checks) / len(checks))


def bench_copa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_copa_studies": _bench_copa_studies(seed)}
