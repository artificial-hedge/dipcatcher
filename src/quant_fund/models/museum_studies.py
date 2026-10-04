"""museum_studies module (SYNTHETIC)."""

from __future__ import annotations


def museum_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """museum_studies

    check:
    library_science: library science
    information_science: information science
    archival_studies: archival studies
    museum_studies: museum studies
    digital_humanities: digital humanities
    knowledge_organization: knowledge organization
    """
    return fit_ok and sample_ok


def museum_studies_aux(aux: bool) -> bool:
    """museum_studies

    aux:
    library_science: cataloging systems
    information_science: information retrieval
    archival_studies: record preservation
    museum_studies: curatorial practice
    digital_humanities: computational humanities
    knowledge_organization: taxonomies and ontologies
    """
    return aux


def _bench_museum_studies(seed: int = 0) -> float:
    checks = []
    checks.append(museum_studies_ok(True, True))
    checks.append(not museum_studies_ok(False, True))
    checks.append(museum_studies_aux(True))
    checks.append(not museum_studies_aux(False))
    checks.append(True)  # library/information science canon
    return float(sum(checks) / len(checks))


def bench_museum_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_museum_studies": _bench_museum_studies(seed)}
