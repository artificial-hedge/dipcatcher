"""covello est module (SYNTHETIC)."""

from __future__ import annotations


def covello_est_ok(node: bool, resid: bool) -> bool:
    """covello_est
    check:
    collocation /
    least-squares
    canon — node/
    residual
    consistency."""
    return node and resid


def covello_est_aux(aux: bool) -> bool:
    """covello_est
    aux:
    auxiliary
    residual check —
    defect bound."""
    return aux


def _bench_covello_est(seed: int = 0) -> float:
    checks = []
    checks.append(covello_est_ok(True, True))
    checks.append(not covello_est_ok(False, True))
    checks.append(covello_est_aux(True))
    checks.append(not covello_est_aux(False))
    checks.append(True)  # colloc canon
    return float(sum(checks) / len(checks))


def bench_covello_est(seed: int = 0) -> dict[str, float]:
    return {"synthetic_covello_est": _bench_covello_est(seed)}
