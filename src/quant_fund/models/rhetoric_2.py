"""rhetoric_2 module (SYNTHETIC)."""

from __future__ import annotations


def rhetoric_2_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """rhetoric_2

    check:
    communication_3: communication
    journalism_3: journalism
    media_studies_3: media studies
    rhetoric_2: rhetoric
    information_science_3: information science
    digital_media_2: digital media
    """
    return fit_ok and sample_ok


def rhetoric_2_aux(aux: bool) -> bool:
    """rhetoric_2

    aux:
    communication_3: messages and channels
    journalism_3: reporting and editing
    media_studies_3: outlets and audiences
    rhetoric_2: persuasion and argument
    information_science_3: indexing and retrieval
    digital_media_2: platforms and streaming
    """
    return aux


def _bench_rhetoric_2(seed: int = 0) -> float:
    checks = []
    checks.append(rhetoric_2_ok(True, True))
    checks.append(not rhetoric_2_ok(False, True))
    checks.append(rhetoric_2_aux(True))
    checks.append(not rhetoric_2_aux(False))
    checks.append(True)  # media canon
    return float(sum(checks) / len(checks))


def bench_rhetoric_2(seed: int = 0) -> dict[str, float]:
    return {"synthetic_rhetoric_2": _bench_rhetoric_2(seed)}
