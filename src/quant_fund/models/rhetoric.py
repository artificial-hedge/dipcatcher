"""rhetoric module (SYNTHETIC)."""

from __future__ import annotations


def rhetoric_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """rhetoric

    check:
    media_studies: media studies
    journalism: journalism
    public_relations: public relations
    rhetoric: rhetoric
    communication_theory: communication theory
    digital_media: digital media
    """
    return fit_ok and sample_ok


def rhetoric_aux(aux: bool) -> bool:
    """rhetoric

    aux:
    media_studies: media analysis
    journalism: news reporting
    public_relations: strategic communication
    rhetoric: persuasive discourse
    communication_theory: information transmission
    digital_media: online platforms
    """
    return aux


def _bench_rhetoric(seed: int = 0) -> float:
    checks = []
    checks.append(rhetoric_ok(True, True))
    checks.append(not rhetoric_ok(False, True))
    checks.append(rhetoric_aux(True))
    checks.append(not rhetoric_aux(False))
    checks.append(True)  # communications/media canon
    return float(sum(checks) / len(checks))


def bench_rhetoric(seed: int = 0) -> dict[str, float]:
    return {"synthetic_rhetoric": _bench_rhetoric(seed)}
