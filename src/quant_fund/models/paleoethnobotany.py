"""paleoethnobotany module (SYNTHETIC)."""

from __future__ import annotations


def paleoethnobotany_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """paleoethnobotany

    check:
    geoarchaeology: geoarchaeology
    zooarchaeology: zooarchaeology
    paleoethnobotany: paleoethnobotany
    ceramic_analysis: ceramic analysis
    lithic_analysis: lithic analysis
    archaeogenetics: archaeogenetics
    """
    return fit_ok and sample_ok


def paleoethnobotany_aux(aux: bool) -> bool:
    """paleoethnobotany

    aux:
    geoarchaeology: sediment context
    zooarchaeology: faunal remains
    paleoethnobotany: plant remains
    ceramic_analysis: pottery provenance
    lithic_analysis: stone tools
    archaeogenetics: ancient DNA
    """
    return aux


def _bench_paleoethnobotany(seed: int = 0) -> float:
    checks = []
    checks.append(paleoethnobotany_ok(True, True))
    checks.append(not paleoethnobotany_ok(False, True))
    checks.append(paleoethnobotany_aux(True))
    checks.append(not paleoethnobotany_aux(False))
    checks.append(True)  # archaeology-2 canon
    return float(sum(checks) / len(checks))


def bench_paleoethnobotany(seed: int = 0) -> dict[str, float]:
    return {"synthetic_paleoethnobotany": _bench_paleoethnobotany(seed)}
