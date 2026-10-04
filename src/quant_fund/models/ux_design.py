"""ux_design module (SYNTHETIC)."""

from __future__ import annotations


def ux_design_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """ux_design

    check:
    ux_design: ux design
    hci_studies: hci studies
    information_architecture: information architecture
    interaction_design: interaction design
    accessibility_studies: accessibility studies
    service_design: service design
    """
    return fit_ok and sample_ok


def ux_design_aux(aux: bool) -> bool:
    """ux_design

    aux:
    ux_design: users and journeys
    hci_studies: interfaces and cognition
    information_architecture: structure and navigation
    interaction_design: gestures and feedback
    accessibility_studies: inclusion and barriers
    service_design: touchpoints and journeys
    """
    return aux


def _bench_ux_design(seed: int = 0) -> float:
    checks = []
    checks.append(ux_design_ok(True, True))
    checks.append(not ux_design_ok(False, True))
    checks.append(ux_design_aux(True))
    checks.append(not ux_design_aux(False))
    checks.append(True)  # ux canon
    return float(sum(checks) / len(checks))


def bench_ux_design(seed: int = 0) -> dict[str, float]:
    return {"synthetic_ux_design": _bench_ux_design(seed)}
