"""political_methodology module (SYNTHETIC)."""

from __future__ import annotations


def political_methodology_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """political_methodology

    check:
    american_politics: american politics
    political_behavior: political behavior
    public_law: public law
    political_methodology: political methodology
    security_studies: security studies
    policy_analysis: policy analysis
    """
    return fit_ok and sample_ok


def political_methodology_aux(aux: bool) -> bool:
    """political_methodology

    aux:
    american_politics: american institutions
    political_behavior: voter behavior
    public_law: courts and law
    political_methodology: quantitative methods
    security_studies: strategic analysis
    policy_analysis: policy evaluation
    """
    return aux


def _bench_political_methodology(seed: int = 0) -> float:
    checks = []
    checks.append(political_methodology_ok(True, True))
    checks.append(not political_methodology_ok(False, True))
    checks.append(political_methodology_aux(True))
    checks.append(not political_methodology_aux(False))
    checks.append(True)  # political-science-2 canon
    return float(sum(checks) / len(checks))


def bench_political_methodology(seed: int = 0) -> dict[str, float]:
    return {"synthetic_political_methodology": _bench_political_methodology(seed)}
