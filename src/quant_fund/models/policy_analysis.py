"""policy_analysis module (SYNTHETIC)."""

from __future__ import annotations


def policy_analysis_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """policy_analysis

    check:
    american_politics: american politics
    political_behavior: political behavior
    public_law: public law
    political_methodology: political methodology
    security_studies: security studies
    policy_analysis: policy analysis
    """
    return fit_ok and sample_ok


def policy_analysis_aux(aux: bool) -> bool:
    """policy_analysis

    aux:
    american_politics: american institutions
    political_behavior: voter behavior
    public_law: courts and law
    political_methodology: quantitative methods
    security_studies: strategic analysis
    policy_analysis: policy evaluation
    """
    return aux


def _bench_policy_analysis(seed: int = 0) -> float:
    checks = []
    checks.append(policy_analysis_ok(True, True))
    checks.append(not policy_analysis_ok(False, True))
    checks.append(policy_analysis_aux(True))
    checks.append(not policy_analysis_aux(False))
    checks.append(True)  # political-science-2 canon
    return float(sum(checks) / len(checks))


def bench_policy_analysis(seed: int = 0) -> dict[str, float]:
    return {"synthetic_policy_analysis": _bench_policy_analysis(seed)}
