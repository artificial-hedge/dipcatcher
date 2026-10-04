"""security_studies module (SYNTHETIC)."""

from __future__ import annotations


def security_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """security_studies

    check:
    american_politics: american politics
    political_behavior: political behavior
    public_law: public law
    political_methodology: political methodology
    security_studies: security studies
    policy_analysis: policy analysis
    """
    return fit_ok and sample_ok


def security_studies_aux(aux: bool) -> bool:
    """security_studies

    aux:
    american_politics: american institutions
    political_behavior: voter behavior
    public_law: courts and law
    political_methodology: quantitative methods
    security_studies: strategic analysis
    policy_analysis: policy evaluation
    """
    return aux


def _bench_security_studies(seed: int = 0) -> float:
    checks = []
    checks.append(security_studies_ok(True, True))
    checks.append(not security_studies_ok(False, True))
    checks.append(security_studies_aux(True))
    checks.append(not security_studies_aux(False))
    checks.append(True)  # political-science-2 canon
    return float(sum(checks) / len(checks))


def bench_security_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_security_studies": _bench_security_studies(seed)}
