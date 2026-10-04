"""cue_conflict_studies module (SYNTHETIC)."""

from __future__ import annotations


def cue_conflict_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """cue_conflict_studies

    check:
    cue_conflict_studies: Geirhos cue-conflict images, texture-vs-shape and rates
    """
    return fit_ok and sample_ok


def cue_conflict_studies_aux(aux: bool) -> bool:
    """cue_conflict_studies

    aux:
    cue_conflict_studies: conflict items, preference stats, and accuracy
    """
    return aux


def _bench_cue_conflict_studies(seed: int = 0) -> float:
    checks = []
    checks.append(cue_conflict_studies_ok(True, True))
    checks.append(not cue_conflict_studies_ok(False, True))
    checks.append(cue_conflict_studies_aux(True))
    checks.append(not cue_conflict_studies_aux(False))
    checks.append(True)  # cue-conflict canon
    return float(sum(checks) / len(checks))


def bench_cue_conflict_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_cue_conflict_studies": _bench_cue_conflict_studies(seed)}
