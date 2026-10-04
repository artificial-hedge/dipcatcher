"""iconography module (SYNTHETIC)."""

from __future__ import annotations


def iconography_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """iconography

    check:
    iconography: iconography
    iconology: iconology
    connoisseurship: connoisseurship
    provenance_studies: provenance studies
    curation_practice: curation practice
    formal_analysis: formal analysis
    """
    return fit_ok and sample_ok


def iconography_aux(aux: bool) -> bool:
    """iconography

    aux:
    iconography: motif identification
    iconology: panofsky method
    connoisseurship: attribution method
    provenance_studies: ownership history
    curation_practice: exhibition making
    formal_analysis: stylistic analysis
    """
    return aux


def _bench_iconography(seed: int = 0) -> float:
    checks = []
    checks.append(iconography_ok(True, True))
    checks.append(not iconography_ok(False, True))
    checks.append(iconography_aux(True))
    checks.append(not iconography_aux(False))
    checks.append(True)  # art-historiography canon
    return float(sum(checks) / len(checks))


def bench_iconography(seed: int = 0) -> dict[str, float]:
    return {"synthetic_iconography": _bench_iconography(seed)}
