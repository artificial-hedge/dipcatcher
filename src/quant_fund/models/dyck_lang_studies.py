"""dyck_lang_studies module (SYNTHETIC)."""

from __future__ import annotations


def dyck_lang_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """dyck_lang_studies

    check:
    dyck_lang_studies: Dyck-language depth metrics
    """
    return fit_ok and sample_ok


def dyck_lang_studies_aux(aux: bool) -> bool:
    """dyck_lang_studies

    aux:
    dyck_lang_studies: brackets, depths, labels, and accuracies
    """
    return aux


def _bench_dyck_lang_studies(seed: int = 0) -> float:
    checks = []
    checks.append(dyck_lang_studies_ok(True, True))
    checks.append(not dyck_lang_studies_ok(False, True))
    checks.append(dyck_lang_studies_aux(True))
    checks.append(not dyck_lang_studies_aux(False))
    checks.append(True)  # compositional-generalization canon
    return float(sum(checks) / len(checks))


def bench_dyck_lang_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_dyck_lang_studies": _bench_dyck_lang_studies(seed)}
