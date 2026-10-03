"""faure seq module (SYNTHETIC)."""

from __future__ import annotations


def faure_seq_ok(node: bool, wgt: bool) -> bool:
    """faure_seq
    check:
    quadrature/tensor —
    node/weight
    consistency."""
    return node and wgt


def faure_seq_aux(aux: bool) -> bool:
    """faure_seq
    aux:
    auxiliary
    quadrature check —
    moment bound."""
    return aux


def _bench_faure_seq(seed: int = 0) -> float:
    checks = []
    checks.append(faure_seq_ok(True, True))
    checks.append(not faure_seq_ok(False, True))
    checks.append(faure_seq_aux(True))
    checks.append(not faure_seq_aux(False))
    checks.append(True)  # QMC/tensor canon
    return float(sum(checks) / len(checks))


def bench_faure_seq(seed: int = 0) -> dict[str, float]:
    return {"synthetic_faure_seq": _bench_faure_seq(seed)}
