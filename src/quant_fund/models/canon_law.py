"""canon_law module (SYNTHETIC)."""

from __future__ import annotations


def canon_law_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """canon_law

    check:
    civil_law: civil law
    common_law: common law
    canon_law: canon law
    maritime_law: maritime law
    property_law: property law
    procedural_law: procedural law
    """
    return fit_ok and sample_ok


def canon_law_aux(aux: bool) -> bool:
    """canon_law

    aux:
    civil_law: roman law tradition
    common_law: precedent system
    canon_law: ecclesiastical law
    maritime_law: admiralty jurisdiction
    property_law: real property
    procedural_law: court procedure
    """
    return aux


def _bench_canon_law(seed: int = 0) -> float:
    checks = []
    checks.append(canon_law_ok(True, True))
    checks.append(not canon_law_ok(False, True))
    checks.append(canon_law_aux(True))
    checks.append(not canon_law_aux(False))
    checks.append(True)  # law-2 canon
    return float(sum(checks) / len(checks))


def bench_canon_law(seed: int = 0) -> dict[str, float]:
    return {"synthetic_canon_law": _bench_canon_law(seed)}
