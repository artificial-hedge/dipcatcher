"""american_politics module (SYNTHETIC)."""

from __future__ import annotations


def american_politics_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """american_politics

    check:
    american_politics: american politics
    political_behavior: political behavior
    public_law: public law
    political_methodology: political methodology
    security_studies: security studies
    policy_analysis: policy analysis
    """
    return fit_ok and sample_ok


def american_politics_aux(aux: bool) -> bool:
    """american_politics

    aux:
    american_politics: american institutions
    political_behavior: voter behavior
    public_law: courts and law
    political_methodology: quantitative methods
    security_studies: strategic analysis
    policy_analysis: policy evaluation
    """
    return aux


def _bench_american_politics(seed: int = 0) -> float:
    checks = []
    checks.append(american_politics_ok(True, True))
    checks.append(not american_politics_ok(False, True))
    checks.append(american_politics_aux(True))
    checks.append(not american_politics_aux(False))
    checks.append(True)  # political-science-2 canon
    return float(sum(checks) / len(checks))


def bench_american_politics(seed: int = 0) -> dict[str, float]:
    return {"synthetic_american_politics": _bench_american_politics(seed)}
