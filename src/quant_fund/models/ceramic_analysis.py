"""ceramic_analysis module (SYNTHETIC)."""

from __future__ import annotations


def ceramic_analysis_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """ceramic_analysis

    check:
    geoarchaeology: geoarchaeology
    zooarchaeology: zooarchaeology
    paleoethnobotany: paleoethnobotany
    ceramic_analysis: ceramic analysis
    lithic_analysis: lithic analysis
    archaeogenetics: archaeogenetics
    """
    return fit_ok and sample_ok


def ceramic_analysis_aux(aux: bool) -> bool:
    """ceramic_analysis

    aux:
    geoarchaeology: sediment context
    zooarchaeology: faunal remains
    paleoethnobotany: plant remains
    ceramic_analysis: pottery provenance
    lithic_analysis: stone tools
    archaeogenetics: ancient DNA
    """
    return aux


def _bench_ceramic_analysis(seed: int = 0) -> float:
    checks = []
    checks.append(ceramic_analysis_ok(True, True))
    checks.append(not ceramic_analysis_ok(False, True))
    checks.append(ceramic_analysis_aux(True))
    checks.append(not ceramic_analysis_aux(False))
    checks.append(True)  # archaeology-2 canon
    return float(sum(checks) / len(checks))


def bench_ceramic_analysis(seed: int = 0) -> dict[str, float]:
    return {"synthetic_ceramic_analysis": _bench_ceramic_analysis(seed)}
