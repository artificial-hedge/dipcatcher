"""political_science_4 module (SYNTHETIC)."""

from __future__ import annotations


def political_science_4_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """political_science_4

    check:
    law_5: law
    political_science_4: political science
    public_administration_2: public administration
    international_relations_2: international relations
    criminology_2: criminology
    military_science_2: military science
    """
    return fit_ok and sample_ok


def political_science_4_aux(aux: bool) -> bool:
    """political_science_4

    aux:
    law_5: statutes and precedents
    political_science_4: institutions and elections
    public_administration_2: agencies and budgets
    international_relations_2: states and treaties
    criminology_2: offenses and enforcement
    military_science_2: strategy and logistics
    """
    return aux


def _bench_political_science_4(seed: int = 0) -> float:
    checks = []
    checks.append(political_science_4_ok(True, True))
    checks.append(not political_science_4_ok(False, True))
    checks.append(political_science_4_aux(True))
    checks.append(not political_science_4_aux(False))
    checks.append(True)  # governance canon
    return float(sum(checks) / len(checks))


def bench_political_science_4(seed: int = 0) -> dict[str, float]:
    return {"synthetic_political_science_4": _bench_political_science_4(seed)}
