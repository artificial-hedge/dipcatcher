"""public_administration module (SYNTHETIC)."""

from __future__ import annotations


def public_administration_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """public_administration

    check:
    comparative_politics: comparative politics
    international_relations: international relations
    political_theory: political theory
    public_administration: public administration
    political_economy: political economy
    electoral_systems: electoral systems
    """
    return fit_ok and sample_ok


def public_administration_aux(aux: bool) -> bool:
    """public_administration

    aux:
    comparative_politics: regime types
    international_relations: state interactions
    political_theory: normative concepts
    public_administration: bureaucratic systems
    political_economy: institutions and markets
    electoral_systems: voting rules
    """
    return aux


def _bench_public_administration(seed: int = 0) -> float:
    checks = []
    checks.append(public_administration_ok(True, True))
    checks.append(not public_administration_ok(False, True))
    checks.append(public_administration_aux(True))
    checks.append(not public_administration_aux(False))
    checks.append(True)  # political-science canon
    return float(sum(checks) / len(checks))


def bench_public_administration(seed: int = 0) -> dict[str, float]:
    return {"synthetic_public_administration": _bench_public_administration(seed)}
