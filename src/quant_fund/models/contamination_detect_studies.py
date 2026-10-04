"""contamination_detect_studies module (SYNTHETIC)."""

from __future__ import annotations


def contamination_detect_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """contamination_detect_studies

    check:
    contamination_detect_studies: overlap and memorization probes/n-grams and test leakage
    """
    return fit_ok and sample_ok


def contamination_detect_studies_aux(aux: bool) -> bool:
    """contamination_detect_studies

    aux:
    contamination_detect_studies: membership inference and canaries/dedup and detection
    """
    return aux


def _bench_contamination_detect_studies(seed: int = 0) -> float:
    checks = []
    checks.append(contamination_detect_studies_ok(True, True))
    checks.append(not contamination_detect_studies_ok(False, True))
    checks.append(contamination_detect_studies_aux(True))
    checks.append(not contamination_detect_studies_aux(False))
    checks.append(True)  # LLM-evaluation canon
    return float(sum(checks) / len(checks))


def bench_contamination_detect_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_contamination_detect_studies": _bench_contamination_detect_studies(seed)}
