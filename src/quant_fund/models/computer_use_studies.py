"""computer_use_studies module (SYNTHETIC)."""

from __future__ import annotations


def computer_use_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """computer_use_studies

    check:
    computer_use_studies: screen grounding and GUI actions/clicks and coordinates
    """
    return fit_ok and sample_ok


def computer_use_studies_aux(aux: bool) -> bool:
    """computer_use_studies

    aux:
    computer_use_studies: VNC-style observation and action space/navigation and OS tasks
    """
    return aux


def _bench_computer_use_studies(seed: int = 0) -> float:
    checks = []
    checks.append(computer_use_studies_ok(True, True))
    checks.append(not computer_use_studies_ok(False, True))
    checks.append(computer_use_studies_aux(True))
    checks.append(not computer_use_studies_aux(False))
    checks.append(True)  # agent-infrastructure canon
    return float(sum(checks) / len(checks))


def bench_computer_use_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_computer_use_studies": _bench_computer_use_studies(seed)}
