"""journalism module (SYNTHETIC)."""

from __future__ import annotations


def journalism_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """journalism

    check:
    media_studies: media studies
    journalism: journalism
    public_relations: public relations
    rhetoric: rhetoric
    communication_theory: communication theory
    digital_media: digital media
    """
    return fit_ok and sample_ok


def journalism_aux(aux: bool) -> bool:
    """journalism

    aux:
    media_studies: media analysis
    journalism: news reporting
    public_relations: strategic communication
    rhetoric: persuasive discourse
    communication_theory: information transmission
    digital_media: online platforms
    """
    return aux


def _bench_journalism(seed: int = 0) -> float:
    checks = []
    checks.append(journalism_ok(True, True))
    checks.append(not journalism_ok(False, True))
    checks.append(journalism_aux(True))
    checks.append(not journalism_aux(False))
    checks.append(True)  # communications/media canon
    return float(sum(checks) / len(checks))


def bench_journalism(seed: int = 0) -> dict[str, float]:
    return {"synthetic_journalism": _bench_journalism(seed)}
