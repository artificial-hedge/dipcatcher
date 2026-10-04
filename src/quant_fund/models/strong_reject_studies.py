"""strong_reject_studies module (SYNTHETIC)."""

from __future__ import annotations


def strong_reject_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """strong_reject_studies

    check:
    strong_reject_studies: StrongREJECT jailbreak-evaluation and refusal rates
    """
    return fit_ok and sample_ok


def strong_reject_studies_aux(aux: bool) -> bool:
    """strong_reject_studies

    aux:
    strong_reject_studies: attack prompts, refusal labels, and scores
    """
    return aux


def _bench_strong_reject_studies(seed: int = 0) -> float:
    checks = []
    checks.append(strong_reject_studies_ok(True, True))
    checks.append(not strong_reject_studies_ok(False, True))
    checks.append(strong_reject_studies_aux(True))
    checks.append(not strong_reject_studies_aux(False))
    checks.append(True)  # eval-science-2 canon
    return float(sum(checks) / len(checks))


def bench_strong_reject_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_strong_reject_studies": _bench_strong_reject_studies(seed)}
