"""connoisseurship module (SYNTHETIC)."""

from __future__ import annotations


def connoisseurship_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """connoisseurship

    check:
    iconography: iconography
    iconology: iconology
    connoisseurship: connoisseurship
    provenance_studies: provenance studies
    curation_practice: curation practice
    formal_analysis: formal analysis
    """
    return fit_ok and sample_ok


def connoisseurship_aux(aux: bool) -> bool:
    """connoisseurship

    aux:
    iconography: motif identification
    iconology: panofsky method
    connoisseurship: attribution method
    provenance_studies: ownership history
    curation_practice: exhibition making
    formal_analysis: stylistic analysis
    """
    return aux


def _bench_connoisseurship(seed: int = 0) -> float:
    checks = []
    checks.append(connoisseurship_ok(True, True))
    checks.append(not connoisseurship_ok(False, True))
    checks.append(connoisseurship_aux(True))
    checks.append(not connoisseurship_aux(False))
    checks.append(True)  # art-historiography canon
    return float(sum(checks) / len(checks))


def bench_connoisseurship(seed: int = 0) -> dict[str, float]:
    return {"synthetic_connoisseurship": _bench_connoisseurship(seed)}
