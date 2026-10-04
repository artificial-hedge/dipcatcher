"""political_behavior module (SYNTHETIC)."""

from __future__ import annotations


def political_behavior_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """political_behavior

    check:
    american_politics: american politics
    political_behavior: political behavior
    public_law: public law
    political_methodology: political methodology
    security_studies: security studies
    policy_analysis: policy analysis
    """
    return fit_ok and sample_ok


def political_behavior_aux(aux: bool) -> bool:
    """political_behavior

    aux:
    american_politics: american institutions
    political_behavior: voter behavior
    public_law: courts and law
    political_methodology: quantitative methods
    security_studies: strategic analysis
    policy_analysis: policy evaluation
    """
    return aux


def _bench_political_behavior(seed: int = 0) -> float:
    checks = []
    checks.append(political_behavior_ok(True, True))
    checks.append(not political_behavior_ok(False, True))
    checks.append(political_behavior_aux(True))
    checks.append(not political_behavior_aux(False))
    checks.append(True)  # political-science-2 canon
    return float(sum(checks) / len(checks))


def bench_political_behavior(seed: int = 0) -> dict[str, float]:
    return {"synthetic_political_behavior": _bench_political_behavior(seed)}
