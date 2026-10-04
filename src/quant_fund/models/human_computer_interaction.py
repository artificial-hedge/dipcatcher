"""human_computer_interaction module (SYNTHETIC)."""

from __future__ import annotations


def human_computer_interaction_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """human_computer_interaction

    check:
    interdisciplinary_studies: interdisciplinary studies
    cognitive_science_2: cognitive science
    futures_studies: futures studies
    complexity_science: complexity science
    systems_science: systems science
    human_computer_interaction: human-computer interaction
    """
    return fit_ok and sample_ok


def human_computer_interaction_aux(aux: bool) -> bool:
    """human_computer_interaction

    aux:
    interdisciplinary_studies: boundaries and synthesis
    cognitive_science_2: minds and computation
    futures_studies: scenarios and foresight
    complexity_science: emergence and adaptation
    systems_science: wholes and feedback
    human_computer_interaction: users and interfaces
    """
    return aux


def _bench_human_computer_interaction(seed: int = 0) -> float:
    checks = []
    checks.append(human_computer_interaction_ok(True, True))
    checks.append(not human_computer_interaction_ok(False, True))
    checks.append(human_computer_interaction_aux(True))
    checks.append(not human_computer_interaction_aux(False))
    checks.append(True)  # interdisciplinary canon
    return float(sum(checks) / len(checks))


def bench_human_computer_interaction(seed: int = 0) -> dict[str, float]:
    return {"synthetic_human_computer_interaction": _bench_human_computer_interaction(seed)}
