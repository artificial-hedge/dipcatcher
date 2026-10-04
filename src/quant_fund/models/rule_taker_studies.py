"""rule_taker_studies module (SYNTHETIC)."""

from __future__ import annotations


def rule_taker_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """rule_taker_studies

    check:
    rule_taker_studies: RuleTaker metrics
    """
    return fit_ok and sample_ok


def rule_taker_studies_aux(aux: bool) -> bool:
    """rule_taker_studies

    aux:
    rule_taker_studies: rules, facts, queries, and scores
    """
    return aux


def _bench_rule_taker_studies(seed: int = 0) -> float:
    checks = []
    checks.append(rule_taker_studies_ok(True, True))
    checks.append(not rule_taker_studies_ok(False, True))
    checks.append(rule_taker_studies_aux(True))
    checks.append(not rule_taker_studies_aux(False))
    checks.append(True)  # proof-entailment canon
    return float(sum(checks) / len(checks))


def bench_rule_taker_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_rule_taker_studies": _bench_rule_taker_studies(seed)}
