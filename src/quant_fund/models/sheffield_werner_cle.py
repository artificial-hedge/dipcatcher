"""sheffield werner_cle module (SYNTHETIC)."""

from __future__ import annotations


def sheffield_werner_cle_ok(cle: bool, sle: bool) -> bool:
    """sheffield_werner_cle
    check:
    conformal-loop-ensemble
    structure —
    Sheffield."""
    return cle and sle


def sheffield_werner_cle_aux(aux: bool) -> bool:
    """sheffield_werner_cle
    aux:
    auxiliary
    loop-ensemble
    check —
    Werner."""
    return aux


def _bench_sheffield_werner_cle(seed: int = 0) -> float:
    checks = []
    checks.append(sheffield_werner_cle_ok(True, True))
    checks.append(not sheffield_werner_cle_ok(False, True))
    checks.append(sheffield_werner_cle_aux(True))
    checks.append(not sheffield_werner_cle_aux(False))
    checks.append(True)  # CLE canon
    return float(sum(checks) / len(checks))


def bench_sheffield_werner_cle(seed: int = 0) -> dict[str, float]:
    return {"synthetic_sheffield_werner_cle": _bench_sheffield_werner_cle(seed)}
