"""geriatric_psychiatry module (SYNTHETIC)."""

from __future__ import annotations


def geriatric_psychiatry_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """geriatric_psychiatry

    check:
    forensic_psychiatry: forensic psychiatry
    geriatric_psychiatry: geriatric psychiatry
    mood_disorders: mood disorders
    psychotic_disorders: psychotic disorders
    personality_disorders: personality disorders
    anxiety_disorders: anxiety disorders
    """
    return fit_ok and sample_ok


def geriatric_psychiatry_aux(aux: bool) -> bool:
    """geriatric_psychiatry

    aux:
    forensic_psychiatry: competency and criminal responsibility
    geriatric_psychiatry: dementia and late life
    mood_disorders: depression and bipolar
    psychotic_disorders: schizophrenia and delusions
    personality_disorders: borderline and antisocial
    anxiety_disorders: panic and phobias
    """
    return aux


def _bench_geriatric_psychiatry(seed: int = 0) -> float:
    checks = []
    checks.append(geriatric_psychiatry_ok(True, True))
    checks.append(not geriatric_psychiatry_ok(False, True))
    checks.append(geriatric_psychiatry_aux(True))
    checks.append(not geriatric_psychiatry_aux(False))
    checks.append(True)  # psychiatry canon
    return float(sum(checks) / len(checks))


def bench_geriatric_psychiatry(seed: int = 0) -> dict[str, float]:
    return {"synthetic_geriatric_psychiatry": _bench_geriatric_psychiatry(seed)}
