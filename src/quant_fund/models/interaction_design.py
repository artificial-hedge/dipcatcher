"""interaction_design module (SYNTHETIC)."""

from __future__ import annotations


def interaction_design_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """interaction_design

    check:
    ux_design: ux design
    hci_studies: hci studies
    information_architecture: information architecture
    interaction_design: interaction design
    accessibility_studies: accessibility studies
    service_design: service design
    """
    return fit_ok and sample_ok


def interaction_design_aux(aux: bool) -> bool:
    """interaction_design

    aux:
    ux_design: users and journeys
    hci_studies: interfaces and cognition
    information_architecture: structure and navigation
    interaction_design: gestures and feedback
    accessibility_studies: inclusion and barriers
    service_design: touchpoints and journeys
    """
    return aux


def _bench_interaction_design(seed: int = 0) -> float:
    checks = []
    checks.append(interaction_design_ok(True, True))
    checks.append(not interaction_design_ok(False, True))
    checks.append(interaction_design_aux(True))
    checks.append(not interaction_design_aux(False))
    checks.append(True)  # ux canon
    return float(sum(checks) / len(checks))


def bench_interaction_design(seed: int = 0) -> dict[str, float]:
    return {"synthetic_interaction_design": _bench_interaction_design(seed)}
