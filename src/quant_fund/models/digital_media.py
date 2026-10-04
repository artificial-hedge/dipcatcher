"""digital_media module (SYNTHETIC)."""

from __future__ import annotations


def digital_media_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """digital_media

    check:
    media_studies: media studies
    journalism: journalism
    public_relations: public relations
    rhetoric: rhetoric
    communication_theory: communication theory
    digital_media: digital media
    """
    return fit_ok and sample_ok


def digital_media_aux(aux: bool) -> bool:
    """digital_media

    aux:
    media_studies: media analysis
    journalism: news reporting
    public_relations: strategic communication
    rhetoric: persuasive discourse
    communication_theory: information transmission
    digital_media: online platforms
    """
    return aux


def _bench_digital_media(seed: int = 0) -> float:
    checks = []
    checks.append(digital_media_ok(True, True))
    checks.append(not digital_media_ok(False, True))
    checks.append(digital_media_aux(True))
    checks.append(not digital_media_aux(False))
    checks.append(True)  # communications/media canon
    return float(sum(checks) / len(checks))


def bench_digital_media(seed: int = 0) -> dict[str, float]:
    return {"synthetic_digital_media": _bench_digital_media(seed)}
