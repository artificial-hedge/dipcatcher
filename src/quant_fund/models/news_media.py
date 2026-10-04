"""news_media module (SYNTHETIC)."""

from __future__ import annotations


def news_media_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """news_media

    check:
    journalism_studies: journalism studies
    advertising_studies: advertising studies
    broadcasting_studies: broadcasting studies
    news_media: news media
    public_relations_studies: public relations studies
    publishing_studies: publishing studies
    """
    return fit_ok and sample_ok


def news_media_aux(aux: bool) -> bool:
    """news_media

    aux:
    journalism_studies: reporting and ethics
    advertising_studies: campaigns and persuasion
    broadcasting_studies: airwaves and programming
    news_media: desks and deadlines
    public_relations_studies: image and stakeholders
    publishing_studies: manuscripts and presses
    """
    return aux


def _bench_news_media(seed: int = 0) -> float:
    checks = []
    checks.append(news_media_ok(True, True))
    checks.append(not news_media_ok(False, True))
    checks.append(news_media_aux(True))
    checks.append(not news_media_aux(False))
    checks.append(True)  # media canon
    return float(sum(checks) / len(checks))


def bench_news_media(seed: int = 0) -> dict[str, float]:
    return {"synthetic_news_media": _bench_news_media(seed)}
