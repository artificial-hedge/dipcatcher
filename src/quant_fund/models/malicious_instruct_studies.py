"""malicious_instruct_studies module (SYNTHETIC)."""

from __future__ import annotations


def malicious_instruct_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """malicious_instruct_studies

    check:
    malicious_instruct_studies: MaliciousInstruct instruction compliance and rates
    """
    return fit_ok and sample_ok


def malicious_instruct_studies_aux(aux: bool) -> bool:
    """malicious_instruct_studies

    aux:
    malicious_instruct_studies: harmful instructions, refusals, and safety rates
    """
    return aux


def _bench_malicious_instruct_studies(seed: int = 0) -> float:
    checks = []
    checks.append(malicious_instruct_studies_ok(True, True))
    checks.append(not malicious_instruct_studies_ok(False, True))
    checks.append(malicious_instruct_studies_aux(True))
    checks.append(not malicious_instruct_studies_aux(False))
    checks.append(True)  # risk-domain canon
    return float(sum(checks) / len(checks))


def bench_malicious_instruct_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_malicious_instruct_studies": _bench_malicious_instruct_studies(seed)}
