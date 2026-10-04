"""information_architecture module (SYNTHETIC)."""

from __future__ import annotations


def information_architecture_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """information_architecture

    check:
    ux_design: ux design
    hci_studies: hci studies
    information_architecture: information architecture
    interaction_design: interaction design
    accessibility_studies: accessibility studies
    service_design: service design
    """
    return fit_ok and sample_ok


def information_architecture_aux(aux: bool) -> bool:
    """information_architecture

    aux:
    ux_design: users and journeys
    hci_studies: interfaces and cognition
    information_architecture: structure and navigation
    interaction_design: gestures and feedback
    accessibility_studies: inclusion and barriers
    service_design: touchpoints and journeys
    """
    return aux


def _bench_information_architecture(seed: int = 0) -> float:
    checks = []
    checks.append(information_architecture_ok(True, True))
    checks.append(not information_architecture_ok(False, True))
    checks.append(information_architecture_aux(True))
    checks.append(not information_architecture_aux(False))
    checks.append(True)  # ux canon
    return float(sum(checks) / len(checks))


def bench_information_architecture(seed: int = 0) -> dict[str, float]:
    return {"synthetic_information_architecture": _bench_information_architecture(seed)}
