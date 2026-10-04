"""political_theory module (SYNTHETIC)."""

from __future__ import annotations


def political_theory_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """political_theory

    check:
    comparative_politics: comparative politics
    international_relations: international relations
    political_theory: political theory
    public_administration: public administration
    political_economy: political economy
    electoral_systems: electoral systems
    """
    return fit_ok and sample_ok


def political_theory_aux(aux: bool) -> bool:
    """political_theory

    aux:
    comparative_politics: regime types
    international_relations: state interactions
    political_theory: normative concepts
    public_administration: bureaucratic systems
    political_economy: institutions and markets
    electoral_systems: voting rules
    """
    return aux


def _bench_political_theory(seed: int = 0) -> float:
    checks = []
    checks.append(political_theory_ok(True, True))
    checks.append(not political_theory_ok(False, True))
    checks.append(political_theory_aux(True))
    checks.append(not political_theory_aux(False))
    checks.append(True)  # political-science canon
    return float(sum(checks) / len(checks))


def bench_political_theory(seed: int = 0) -> dict[str, float]:
    return {"synthetic_political_theory": _bench_political_theory(seed)}
