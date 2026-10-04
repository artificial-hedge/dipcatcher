"""decontaminate_studies module (SYNTHETIC)."""

from __future__ import annotations


def decontaminate_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """decontaminate_studies

    check:
    decontaminate_studies: Test-set decontamination detection metrics
    """
    return fit_ok and sample_ok


def decontaminate_studies_aux(aux: bool) -> bool:
    """decontaminate_studies

    aux:
    decontaminate_studies: corpora, n-grams, matches, and contamination scores
    """
    return aux


def _bench_decontaminate_studies(seed: int = 0) -> float:
    checks = []
    checks.append(decontaminate_studies_ok(True, True))
    checks.append(not decontaminate_studies_ok(False, True))
    checks.append(decontaminate_studies_aux(True))
    checks.append(not decontaminate_studies_aux(False))
    checks.append(True)  # eval-tooling canon
    return float(sum(checks) / len(checks))


def bench_decontaminate_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_decontaminate_studies": _bench_decontaminate_studies(seed)}
