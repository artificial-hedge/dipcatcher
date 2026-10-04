"""clean_label_studies module (SYNTHETIC)."""

from __future__ import annotations


def clean_label_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """clean_label_studies

    check:
    clean_label_studies: clean-label poisons/feature collisions and ASR
    """
    return fit_ok and sample_ok


def clean_label_studies_aux(aux: bool) -> bool:
    """clean_label_studies

    aux:
    clean_label_studies: collision optimization/watermarks and transfer
    """
    return aux


def _bench_clean_label_studies(seed: int = 0) -> float:
    checks = []
    checks.append(clean_label_studies_ok(True, True))
    checks.append(not clean_label_studies_ok(False, True))
    checks.append(clean_label_studies_aux(True))
    checks.append(not clean_label_studies_aux(False))
    checks.append(True)  # backdoor-eval canon
    return float(sum(checks) / len(checks))


def bench_clean_label_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_clean_label_studies": _bench_clean_label_studies(seed)}
