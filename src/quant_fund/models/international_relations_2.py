"""international_relations_2 module (SYNTHETIC)."""

from __future__ import annotations


def international_relations_2_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """international_relations_2

    check:
    law_5: law
    political_science_4: political science
    public_administration_2: public administration
    international_relations_2: international relations
    criminology_2: criminology
    military_science_2: military science
    """
    return fit_ok and sample_ok


def international_relations_2_aux(aux: bool) -> bool:
    """international_relations_2

    aux:
    law_5: statutes and precedents
    political_science_4: institutions and elections
    public_administration_2: agencies and budgets
    international_relations_2: states and treaties
    criminology_2: offenses and enforcement
    military_science_2: strategy and logistics
    """
    return aux


def _bench_international_relations_2(seed: int = 0) -> float:
    checks = []
    checks.append(international_relations_2_ok(True, True))
    checks.append(not international_relations_2_ok(False, True))
    checks.append(international_relations_2_aux(True))
    checks.append(not international_relations_2_aux(False))
    checks.append(True)  # governance canon
    return float(sum(checks) / len(checks))


def bench_international_relations_2(seed: int = 0) -> dict[str, float]:
    return {"synthetic_international_relations_2": _bench_international_relations_2(seed)}
