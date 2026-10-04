"""media_studies module (SYNTHETIC)."""

from __future__ import annotations


def media_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """media_studies

    check:
    media_studies: media studies
    journalism: journalism
    public_relations: public relations
    rhetoric: rhetoric
    communication_theory: communication theory
    digital_media: digital media
    """
    return fit_ok and sample_ok


def media_studies_aux(aux: bool) -> bool:
    """media_studies

    aux:
    media_studies: media analysis
    journalism: news reporting
    public_relations: strategic communication
    rhetoric: persuasive discourse
    communication_theory: information transmission
    digital_media: online platforms
    """
    return aux


def _bench_media_studies(seed: int = 0) -> float:
    checks = []
    checks.append(media_studies_ok(True, True))
    checks.append(not media_studies_ok(False, True))
    checks.append(media_studies_aux(True))
    checks.append(not media_studies_aux(False))
    checks.append(True)  # communications/media canon
    return float(sum(checks) / len(checks))


def bench_media_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_media_studies": _bench_media_studies(seed)}
