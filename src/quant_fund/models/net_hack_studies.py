"""net_hack_studies module (SYNTHETIC)."""

from __future__ import annotations


def net_hack_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """net_hack_studies

    check:
    net_hack_studies: NetHack metrics
    """
    return fit_ok and sample_ok


def net_hack_studies_aux(aux: bool) -> bool:
    """net_hack_studies

    aux:
    net_hack_studies: episodes, actions, rewards, and scores
    """
    return aux


def _bench_net_hack_studies(seed: int = 0) -> float:
    checks = []
    checks.append(net_hack_studies_ok(True, True))
    checks.append(not net_hack_studies_ok(False, True))
    checks.append(net_hack_studies_aux(True))
    checks.append(not net_hack_studies_aux(False))
    checks.append(True)  # MCP-web canon
    return float(sum(checks) / len(checks))


def bench_net_hack_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_net_hack_studies": _bench_net_hack_studies(seed)}
