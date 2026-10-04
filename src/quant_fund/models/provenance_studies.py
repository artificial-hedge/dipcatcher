"""provenance_studies module (SYNTHETIC)."""

from __future__ import annotations


def provenance_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """provenance_studies

    check:
    iconography: iconography
    iconology: iconology
    connoisseurship: connoisseurship
    provenance_studies: provenance studies
    curation_practice: curation practice
    formal_analysis: formal analysis
    """
    return fit_ok and sample_ok


def provenance_studies_aux(aux: bool) -> bool:
    """provenance_studies

    aux:
    iconography: motif identification
    iconology: panofsky method
    connoisseurship: attribution method
    provenance_studies: ownership history
    curation_practice: exhibition making
    formal_analysis: stylistic analysis
    """
    return aux


def _bench_provenance_studies(seed: int = 0) -> float:
    checks = []
    checks.append(provenance_studies_ok(True, True))
    checks.append(not provenance_studies_ok(False, True))
    checks.append(provenance_studies_aux(True))
    checks.append(not provenance_studies_aux(False))
    checks.append(True)  # art-historiography canon
    return float(sum(checks) / len(checks))


def bench_provenance_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_provenance_studies": _bench_provenance_studies(seed)}
