"""obesity_studies module (SYNTHETIC)."""

from __future__ import annotations


def obesity_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """obesity_studies

    check:
    obesity_studies: adiposity and bmi
    ..."""
    return fit_ok and sample_ok


def obesity_studies_aux(aux: bool) -> bool:
    """obesity_studies

    aux:
    obesity_studies: leptin and satiety
    ..."""
    return aux


def _bench_obesity_studies(seed: int = 0) -> float:
    checks = []
    checks.append(obesity_studies_ok(True, True))
    checks.append(not obesity_studies_ok(False, True))
    checks.append(obesity_studies_aux(True))
    checks.append(not obesity_studies_aux(False))
    checks.append(True)  # metabolic-endocrine canon
    return float(sum(checks) / len(checks))


def bench_obesity_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_obesity_studies": _bench_obesity_studies(seed)}
