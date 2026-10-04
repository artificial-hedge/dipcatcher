"""cognitive_science_2 module (SYNTHETIC)."""

from __future__ import annotations


def cognitive_science_2_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """cognitive_science_2

    check:
    interdisciplinary_studies: interdisciplinary studies
    cognitive_science_2: cognitive science
    futures_studies: futures studies
    complexity_science: complexity science
    systems_science: systems science
    human_computer_interaction: human-computer interaction
    """
    return fit_ok and sample_ok


def cognitive_science_2_aux(aux: bool) -> bool:
    """cognitive_science_2

    aux:
    interdisciplinary_studies: boundaries and synthesis
    cognitive_science_2: minds and computation
    futures_studies: scenarios and foresight
    complexity_science: emergence and adaptation
    systems_science: wholes and feedback
    human_computer_interaction: users and interfaces
    """
    return aux


def _bench_cognitive_science_2(seed: int = 0) -> float:
    checks = []
    checks.append(cognitive_science_2_ok(True, True))
    checks.append(not cognitive_science_2_ok(False, True))
    checks.append(cognitive_science_2_aux(True))
    checks.append(not cognitive_science_2_aux(False))
    checks.append(True)  # interdisciplinary canon
    return float(sum(checks) / len(checks))


def bench_cognitive_science_2(seed: int = 0) -> dict[str, float]:
    return {"synthetic_cognitive_science_2": _bench_cognitive_science_2(seed)}
