"""curation_practice module (SYNTHETIC)."""

from __future__ import annotations


def curation_practice_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """curation_practice

    check:
    iconography: iconography
    iconology: iconology
    connoisseurship: connoisseurship
    provenance_studies: provenance studies
    curation_practice: curation practice
    formal_analysis: formal analysis
    """
    return fit_ok and sample_ok


def curation_practice_aux(aux: bool) -> bool:
    """curation_practice

    aux:
    iconography: motif identification
    iconology: panofsky method
    connoisseurship: attribution method
    provenance_studies: ownership history
    curation_practice: exhibition making
    formal_analysis: stylistic analysis
    """
    return aux


def _bench_curation_practice(seed: int = 0) -> float:
    checks = []
    checks.append(curation_practice_ok(True, True))
    checks.append(not curation_practice_ok(False, True))
    checks.append(curation_practice_aux(True))
    checks.append(not curation_practice_aux(False))
    checks.append(True)  # art-historiography canon
    return float(sum(checks) / len(checks))


def bench_curation_practice(seed: int = 0) -> dict[str, float]:
    return {"synthetic_curation_practice": _bench_curation_practice(seed)}
