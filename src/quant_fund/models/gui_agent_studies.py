"""gui_agent_studies module (SYNTHETIC)."""

from __future__ import annotations


def gui_agent_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """gui_agent_studies

    check:
    gui_agent_studies: screen grounding and action prediction/elements and clicks
    """
    return fit_ok and sample_ok


def gui_agent_studies_aux(aux: bool) -> bool:
    """gui_agent_studies

    aux:
    gui_agent_studies: UI navigation policies and goals/tasks and steps
    """
    return aux


def _bench_gui_agent_studies(seed: int = 0) -> float:
    checks = []
    checks.append(gui_agent_studies_ok(True, True))
    checks.append(not gui_agent_studies_ok(False, True))
    checks.append(gui_agent_studies_aux(True))
    checks.append(not gui_agent_studies_aux(False))
    checks.append(True)  # multimodal-2 canon
    return float(sum(checks) / len(checks))


def bench_gui_agent_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_gui_agent_studies": _bench_gui_agent_studies(seed)}
