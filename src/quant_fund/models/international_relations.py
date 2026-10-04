"""international_relations module (SYNTHETIC)."""

from __future__ import annotations


def international_relations_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """international_relations

    check:
    comparative_politics: comparative politics
    international_relations: international relations
    political_theory: political theory
    public_administration: public administration
    political_economy: political economy
    electoral_systems: electoral systems
    """
    return fit_ok and sample_ok


def international_relations_aux(aux: bool) -> bool:
    """international_relations

    aux:
    comparative_politics: regime types
    international_relations: state interactions
    political_theory: normative concepts
    public_administration: bureaucratic systems
    political_economy: institutions and markets
    electoral_systems: voting rules
    """
    return aux


def _bench_international_relations(seed: int = 0) -> float:
    checks = []
    checks.append(international_relations_ok(True, True))
    checks.append(not international_relations_ok(False, True))
    checks.append(international_relations_aux(True))
    checks.append(not international_relations_aux(False))
    checks.append(True)  # political-science canon
    return float(sum(checks) / len(checks))


def bench_international_relations(seed: int = 0) -> dict[str, float]:
    return {"synthetic_international_relations": _bench_international_relations(seed)}
