"""experimental_archaeology module (SYNTHETIC)."""

from __future__ import annotations


def experimental_archaeology_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """experimental_archaeology

    check:
    field_archaeology: field archaeology
    archaeometry: archaeometry
    bioarchaeology: bioarchaeology
    underwater_archaeology: underwater archaeology
    landscape_archaeology: landscape archaeology
    experimental_archaeology: experimental archaeology
    """
    return fit_ok and sample_ok


def experimental_archaeology_aux(aux: bool) -> bool:
    """experimental_archaeology

    aux:
    field_archaeology: excavation methods
    archaeometry: archaeological dating
    bioarchaeology: human remains
    underwater_archaeology: submerged sites
    landscape_archaeology: landscape survey
    experimental_archaeology: replication studies
    """
    return aux


def _bench_experimental_archaeology(seed: int = 0) -> float:
    checks = []
    checks.append(experimental_archaeology_ok(True, True))
    checks.append(not experimental_archaeology_ok(False, True))
    checks.append(experimental_archaeology_aux(True))
    checks.append(not experimental_archaeology_aux(False))
    checks.append(True)  # archaeology canon
    return float(sum(checks) / len(checks))


def bench_experimental_archaeology(seed: int = 0) -> dict[str, float]:
    return {"synthetic_experimental_archaeology": _bench_experimental_archaeology(seed)}
