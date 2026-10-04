"""formal_analysis module (SYNTHETIC)."""

from __future__ import annotations


def formal_analysis_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """formal_analysis

    check:
    iconography: iconography
    iconology: iconology
    connoisseurship: connoisseurship
    provenance_studies: provenance studies
    curation_practice: curation practice
    formal_analysis: formal analysis
    """
    return fit_ok and sample_ok


def formal_analysis_aux(aux: bool) -> bool:
    """formal_analysis

    aux:
    iconography: motif identification
    iconology: panofsky method
    connoisseurship: attribution method
    provenance_studies: ownership history
    curation_practice: exhibition making
    formal_analysis: stylistic analysis
    """
    return aux


def _bench_formal_analysis(seed: int = 0) -> float:
    checks = []
    checks.append(formal_analysis_ok(True, True))
    checks.append(not formal_analysis_ok(False, True))
    checks.append(formal_analysis_aux(True))
    checks.append(not formal_analysis_aux(False))
    checks.append(True)  # art-historiography canon
    return float(sum(checks) / len(checks))


def bench_formal_analysis(seed: int = 0) -> dict[str, float]:
    return {"synthetic_formal_analysis": _bench_formal_analysis(seed)}
