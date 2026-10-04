"""moral_stories_studies module (SYNTHETIC)."""

from __future__ import annotations


def moral_stories_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """moral_stories_studies

    check:
    moral_stories_studies: Moral Stories ethics metrics
    """
    return fit_ok and sample_ok


def moral_stories_studies_aux(aux: bool) -> bool:
    """moral_stories_studies

    aux:
    moral_stories_studies: stories, actions, judgments, and scores
    """
    return aux


def _bench_moral_stories_studies(seed: int = 0) -> float:
    checks = []
    checks.append(moral_stories_studies_ok(True, True))
    checks.append(not moral_stories_studies_ok(False, True))
    checks.append(moral_stories_studies_aux(True))
    checks.append(not moral_stories_studies_aux(False))
    checks.append(True)  # social-reasoning canon
    return float(sum(checks) / len(checks))


def bench_moral_stories_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_moral_stories_studies": _bench_moral_stories_studies(seed)}
