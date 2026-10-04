"""multipl_e_studies module (SYNTHETIC)."""

from __future__ import annotations


def multipl_e_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """multipl_e_studies

    check:
    multipl_e_studies: MultiPL-E translation pass@1 across languages
    """
    return fit_ok and sample_ok


def multipl_e_studies_aux(aux: bool) -> bool:
    """multipl_e_studies

    aux:
    multipl_e_studies: python tasks, translations, and pass rates
    """
    return aux


def _bench_multipl_e_studies(seed: int = 0) -> float:
    checks = []
    checks.append(multipl_e_studies_ok(True, True))
    checks.append(not multipl_e_studies_ok(False, True))
    checks.append(multipl_e_studies_aux(True))
    checks.append(not multipl_e_studies_aux(False))
    checks.append(True)  # code-eval-2 canon
    return float(sum(checks) / len(checks))


def bench_multipl_e_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_multipl_e_studies": _bench_multipl_e_studies(seed)}
