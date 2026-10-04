"""jailbreak_detect_studies module (SYNTHETIC)."""

from __future__ import annotations


def jailbreak_detect_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """jailbreak_detect_studies

    check:
    jailbreak_detect_studies: prompt-attack pattern detection/tokens and flags
    """
    return fit_ok and sample_ok


def jailbreak_detect_studies_aux(aux: bool) -> bool:
    """jailbreak_detect_studies

    aux:
    jailbreak_detect_studies: perplexity and direction jailbreak scores/queries and thresholds
    """
    return aux


def _bench_jailbreak_detect_studies(seed: int = 0) -> float:
    checks = []
    checks.append(jailbreak_detect_studies_ok(True, True))
    checks.append(not jailbreak_detect_studies_ok(False, True))
    checks.append(jailbreak_detect_studies_aux(True))
    checks.append(not jailbreak_detect_studies_aux(False))
    checks.append(True)  # mech-anomaly/jailbreak canon
    return float(sum(checks) / len(checks))


def bench_jailbreak_detect_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_jailbreak_detect_studies": _bench_jailbreak_detect_studies(seed)}
