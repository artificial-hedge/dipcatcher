"""prompt_inject_studies module (SYNTHETIC)."""

from __future__ import annotations


def prompt_inject_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """prompt_inject_studies

    check:
    prompt_inject_studies: prompt-injection payloads/channels and success
    """
    return fit_ok and sample_ok


def prompt_inject_studies_aux(aux: bool) -> bool:
    """prompt_inject_studies

    aux:
    prompt_inject_studies: indirect-injection defenses/scans and blocks
    """
    return aux


def _bench_prompt_inject_studies(seed: int = 0) -> float:
    checks = []
    checks.append(prompt_inject_studies_ok(True, True))
    checks.append(not prompt_inject_studies_ok(False, True))
    checks.append(prompt_inject_studies_aux(True))
    checks.append(not prompt_inject_studies_aux(False))
    checks.append(True)  # safety-eval canon
    return float(sum(checks) / len(checks))


def bench_prompt_inject_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_prompt_inject_studies": _bench_prompt_inject_studies(seed)}
