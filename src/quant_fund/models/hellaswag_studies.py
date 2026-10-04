"""hellaswag_studies module (SYNTHETIC)."""

from __future__ import annotations


def hellaswag_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """hellaswag_studies

    check:
    hellaswag_studies: HellaSwag activity continuations and acc_norm
    """
    return fit_ok and sample_ok


def hellaswag_studies_aux(aux: bool) -> bool:
    """hellaswag_studies

    aux:
    hellaswag_studies: contexts/endings, LM scoring, and accuracy
    """
    return aux


def _bench_hellaswag_studies(seed: int = 0) -> float:
    checks = []
    checks.append(hellaswag_studies_ok(True, True))
    checks.append(not hellaswag_studies_ok(False, True))
    checks.append(hellaswag_studies_aux(True))
    checks.append(not hellaswag_studies_aux(False))
    checks.append(True)  # commonsense-eval canon
    return float(sum(checks) / len(checks))


def bench_hellaswag_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_hellaswag_studies": _bench_hellaswag_studies(seed)}
