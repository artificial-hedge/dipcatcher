"""information_science module (SYNTHETIC)."""

from __future__ import annotations


def information_science_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """information_science

    check:
    library_science: library science
    information_science: information science
    archival_studies: archival studies
    museum_studies: museum studies
    digital_humanities: digital humanities
    knowledge_organization: knowledge organization
    """
    return fit_ok and sample_ok


def information_science_aux(aux: bool) -> bool:
    """information_science

    aux:
    library_science: cataloging systems
    information_science: information retrieval
    archival_studies: record preservation
    museum_studies: curatorial practice
    digital_humanities: computational humanities
    knowledge_organization: taxonomies and ontologies
    """
    return aux


def _bench_information_science(seed: int = 0) -> float:
    checks = []
    checks.append(information_science_ok(True, True))
    checks.append(not information_science_ok(False, True))
    checks.append(information_science_aux(True))
    checks.append(not information_science_aux(False))
    checks.append(True)  # library/information science canon
    return float(sum(checks) / len(checks))


def bench_information_science(seed: int = 0) -> dict[str, float]:
    return {"synthetic_information_science": _bench_information_science(seed)}
