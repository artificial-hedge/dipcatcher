"""choreography_2 module (SYNTHETIC)."""

from __future__ import annotations


def choreography_2_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """choreography_2

    check:
    ballet_studies: ballet studies
    choreography_2: choreography
    dance_pedagogy: dance pedagogy
    somatic_practices: somatic practices
    dance_science: dance science
    movement_studies: movement studies
    """
    return fit_ok and sample_ok


def choreography_2_aux(aux: bool) -> bool:
    """choreography_2

    aux:
    ballet_studies: technique and repertoire
    choreography_2: movement and composition
    dance_pedagogy: teaching and curriculum
    somatic_practices: body and awareness
    dance_science: performance and conditioning
    movement_studies: gesture and embodiment
    """
    return aux


def _bench_choreography_2(seed: int = 0) -> float:
    checks = []
    checks.append(choreography_2_ok(True, True))
    checks.append(not choreography_2_ok(False, True))
    checks.append(choreography_2_aux(True))
    checks.append(not choreography_2_aux(False))
    checks.append(True)  # dance canon
    return float(sum(checks) / len(checks))


def bench_choreography_2(seed: int = 0) -> dict[str, float]:
    return {"synthetic_choreography_2": _bench_choreography_2(seed)}
