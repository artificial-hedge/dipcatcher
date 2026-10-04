"""communication_theory module (SYNTHETIC)."""

from __future__ import annotations


def communication_theory_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """communication_theory

    check:
    media_studies: media studies
    journalism: journalism
    public_relations: public relations
    rhetoric: rhetoric
    communication_theory: communication theory
    digital_media: digital media
    """
    return fit_ok and sample_ok


def communication_theory_aux(aux: bool) -> bool:
    """communication_theory

    aux:
    media_studies: media analysis
    journalism: news reporting
    public_relations: strategic communication
    rhetoric: persuasive discourse
    communication_theory: information transmission
    digital_media: online platforms
    """
    return aux


def _bench_communication_theory(seed: int = 0) -> float:
    checks = []
    checks.append(communication_theory_ok(True, True))
    checks.append(not communication_theory_ok(False, True))
    checks.append(communication_theory_aux(True))
    checks.append(not communication_theory_aux(False))
    checks.append(True)  # communications/media canon
    return float(sum(checks) / len(checks))


def bench_communication_theory(seed: int = 0) -> dict[str, float]:
    return {"synthetic_communication_theory": _bench_communication_theory(seed)}
