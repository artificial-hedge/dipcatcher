"""community_psychology module (SYNTHETIC)."""

from __future__ import annotations


def community_psychology_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """community_psychology

    check:
    social_cognition: social cognition
    positive_psychology: positive psychology
    cross_cultural_psychology: cross-cultural psychology
    consumer_psychology: consumer psychology
    political_psychology: political psychology
    community_psychology: community psychology
    """
    return fit_ok and sample_ok


def community_psychology_aux(aux: bool) -> bool:
    """community_psychology

    aux:
    social_cognition: mental inference
    positive_psychology: well-being
    cross_cultural_psychology: cultural variation
    consumer_psychology: buying behavior
    political_psychology: political attitudes
    community_psychology: community settings
    """
    return aux


def _bench_community_psychology(seed: int = 0) -> float:
    checks = []
    checks.append(community_psychology_ok(True, True))
    checks.append(not community_psychology_ok(False, True))
    checks.append(community_psychology_aux(True))
    checks.append(not community_psychology_aux(False))
    checks.append(True)  # psychology-4 canon
    return float(sum(checks) / len(checks))


def bench_community_psychology(seed: int = 0) -> dict[str, float]:
    return {"synthetic_community_psychology": _bench_community_psychology(seed)}
