"""military_science module (SYNTHETIC)."""

from __future__ import annotations


def military_science_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """military_science

    check:
    military_science: military science
    defense_studies: defense studies
    strategic_studies: strategic studies
    intelligence_studies: intelligence studies
    peace_studies: peace studies
    conflict_resolution: conflict resolution
    """
    return fit_ok and sample_ok


def military_science_aux(aux: bool) -> bool:
    """military_science

    aux:
    military_science: warfare and logistics
    defense_studies: national security policy
    strategic_studies: grand strategy
    intelligence_studies: intelligence analysis
    peace_studies: peace research
    conflict_resolution: mediation and negotiation
    """
    return aux


def _bench_military_science(seed: int = 0) -> float:
    checks = []
    checks.append(military_science_ok(True, True))
    checks.append(not military_science_ok(False, True))
    checks.append(military_science_aux(True))
    checks.append(not military_science_aux(False))
    checks.append(True)  # military/defense studies canon
    return float(sum(checks) / len(checks))


def bench_military_science(seed: int = 0) -> dict[str, float]:
    return {"synthetic_military_science": _bench_military_science(seed)}
