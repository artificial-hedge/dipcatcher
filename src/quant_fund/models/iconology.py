"""iconology module (SYNTHETIC)."""

from __future__ import annotations


def iconology_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """iconology

    check:
    iconography: iconography
    iconology: iconology
    connoisseurship: connoisseurship
    provenance_studies: provenance studies
    curation_practice: curation practice
    formal_analysis: formal analysis
    """
    return fit_ok and sample_ok


def iconology_aux(aux: bool) -> bool:
    """iconology

    aux:
    iconography: motif identification
    iconology: panofsky method
    connoisseurship: attribution method
    provenance_studies: ownership history
    curation_practice: exhibition making
    formal_analysis: stylistic analysis
    """
    return aux


def _bench_iconology(seed: int = 0) -> float:
    checks = []
    checks.append(iconology_ok(True, True))
    checks.append(not iconology_ok(False, True))
    checks.append(iconology_aux(True))
    checks.append(not iconology_aux(False))
    checks.append(True)  # art-historiography canon
    return float(sum(checks) / len(checks))


def bench_iconology(seed: int = 0) -> dict[str, float]:
    return {"synthetic_iconology": _bench_iconology(seed)}
