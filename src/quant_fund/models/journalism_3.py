"""journalism_3 module (SYNTHETIC)."""

from __future__ import annotations


def journalism_3_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """journalism_3

    check:
    communication_3: communication
    journalism_3: journalism
    media_studies_3: media studies
    rhetoric_2: rhetoric
    information_science_3: information science
    digital_media_2: digital media
    """
    return fit_ok and sample_ok


def journalism_3_aux(aux: bool) -> bool:
    """journalism_3

    aux:
    communication_3: messages and channels
    journalism_3: reporting and editing
    media_studies_3: outlets and audiences
    rhetoric_2: persuasion and argument
    information_science_3: indexing and retrieval
    digital_media_2: platforms and streaming
    """
    return aux


def _bench_journalism_3(seed: int = 0) -> float:
    checks = []
    checks.append(journalism_3_ok(True, True))
    checks.append(not journalism_3_ok(False, True))
    checks.append(journalism_3_aux(True))
    checks.append(not journalism_3_aux(False))
    checks.append(True)  # media canon
    return float(sum(checks) / len(checks))


def bench_journalism_3(seed: int = 0) -> dict[str, float]:
    return {"synthetic_journalism_3": _bench_journalism_3(seed)}
