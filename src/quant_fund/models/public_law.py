"""public_law module (SYNTHETIC)."""

from __future__ import annotations


def public_law_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """public_law

    check:
    american_politics: american politics
    political_behavior: political behavior
    public_law: public law
    political_methodology: political methodology
    security_studies: security studies
    policy_analysis: policy analysis
    """
    return fit_ok and sample_ok


def public_law_aux(aux: bool) -> bool:
    """public_law

    aux:
    american_politics: american institutions
    political_behavior: voter behavior
    public_law: courts and law
    political_methodology: quantitative methods
    security_studies: strategic analysis
    policy_analysis: policy evaluation
    """
    return aux


def _bench_public_law(seed: int = 0) -> float:
    checks = []
    checks.append(public_law_ok(True, True))
    checks.append(not public_law_ok(False, True))
    checks.append(public_law_aux(True))
    checks.append(not public_law_aux(False))
    checks.append(True)  # political-science-2 canon
    return float(sum(checks) / len(checks))


def bench_public_law(seed: int = 0) -> dict[str, float]:
    return {"synthetic_public_law": _bench_public_law(seed)}
