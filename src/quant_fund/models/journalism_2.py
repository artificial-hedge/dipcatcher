"""journalism_2 module (SYNTHETIC)."""

from __future__ import annotations


def journalism_2_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """journalism_2

    check:
    education_5: education
    communication_studies_2: communication studies
    media_studies_2: media studies
    journalism_2: journalism
    library_science_2: library science
    information_science_2: information science
    """
    return fit_ok and sample_ok


def journalism_2_aux(aux: bool) -> bool:
    """journalism_2

    aux:
    education_5: pedagogy and curricula
    communication_studies_2: messages and audiences
    media_studies_2: platforms and content
    journalism_2: reporting and verification
    library_science_2: catalogs and collections
    information_science_2: retrieval and metadata
    """
    return aux


def _bench_journalism_2(seed: int = 0) -> float:
    checks = []
    checks.append(journalism_2_ok(True, True))
    checks.append(not journalism_2_ok(False, True))
    checks.append(journalism_2_aux(True))
    checks.append(not journalism_2_aux(False))
    checks.append(True)  # communication canon
    return float(sum(checks) / len(checks))


def bench_journalism_2(seed: int = 0) -> dict[str, float]:
    return {"synthetic_journalism_2": _bench_journalism_2(seed)}
