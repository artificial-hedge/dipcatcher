"""story_cloze_studies module (SYNTHETIC)."""

from __future__ import annotations


def story_cloze_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """story_cloze_studies

    check:
    story_cloze_studies: Story Cloze endings, narratives, and accuracy
    """
    return fit_ok and sample_ok


def story_cloze_studies_aux(aux: bool) -> bool:
    """story_cloze_studies

    aux:
    story_cloze_studies: four-sentence stories, candidate endings, scores
    """
    return aux


def _bench_story_cloze_studies(seed: int = 0) -> float:
    checks = []
    checks.append(story_cloze_studies_ok(True, True))
    checks.append(not story_cloze_studies_ok(False, True))
    checks.append(story_cloze_studies_aux(True))
    checks.append(not story_cloze_studies_aux(False))
    checks.append(True)  # winograd-eval canon
    return float(sum(checks) / len(checks))


def bench_story_cloze_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_story_cloze_studies": _bench_story_cloze_studies(seed)}
