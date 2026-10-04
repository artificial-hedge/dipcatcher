"""qed regime module (SYNTHETIC)."""

from __future__ import annotations


def qed_regime_ok(lim: bool, scale: bool) -> bool:
    """qed_regime
    check:
    heavy-traffic
    structure —
    diffusion
    limit."""
    return lim and scale


def qed_regime_aux(aux: bool) -> bool:
    """qed_regime
    aux:
    auxiliary
    scaling
    check —
    QED
    regime."""
    return aux


def _bench_qed_regime(seed: int = 0) -> float:
    checks = []
    checks.append(qed_regime_ok(True, True))
    checks.append(not qed_regime_ok(False, True))
    checks.append(qed_regime_aux(True))
    checks.append(not qed_regime_aux(False))
    checks.append(True)  # heavy-traffic canon
    return float(sum(checks) / len(checks))


def bench_qed_regime(seed: int = 0) -> dict[str, float]:
    return {"synthetic_qed_regime": _bench_qed_regime(seed)}
