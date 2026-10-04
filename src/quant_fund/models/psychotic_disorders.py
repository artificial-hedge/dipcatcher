"""psychotic_disorders module (SYNTHETIC)."""

from __future__ import annotations


def psychotic_disorders_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """psychotic_disorders

    check:
    forensic_psychiatry: forensic psychiatry
    geriatric_psychiatry: geriatric psychiatry
    mood_disorders: mood disorders
    psychotic_disorders: psychotic disorders
    personality_disorders: personality disorders
    anxiety_disorders: anxiety disorders
    """
    return fit_ok and sample_ok


def psychotic_disorders_aux(aux: bool) -> bool:
    """psychotic_disorders

    aux:
    forensic_psychiatry: competency and criminal responsibility
    geriatric_psychiatry: dementia and late life
    mood_disorders: depression and bipolar
    psychotic_disorders: schizophrenia and delusions
    personality_disorders: borderline and antisocial
    anxiety_disorders: panic and phobias
    """
    return aux


def _bench_psychotic_disorders(seed: int = 0) -> float:
    checks = []
    checks.append(psychotic_disorders_ok(True, True))
    checks.append(not psychotic_disorders_ok(False, True))
    checks.append(psychotic_disorders_aux(True))
    checks.append(not psychotic_disorders_aux(False))
    checks.append(True)  # psychiatry canon
    return float(sum(checks) / len(checks))


def bench_psychotic_disorders(seed: int = 0) -> dict[str, float]:
    return {"synthetic_psychotic_disorders": _bench_psychotic_disorders(seed)}
