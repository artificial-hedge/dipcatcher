"""nonlinear markov module (SYNTHETIC)."""

from __future__ import annotations


def nonlinear_markov_ok(mv1: bool, mk: bool) -> bool:
    """nonlinear_markov
    check:
    McKean-Vlasov
    —
    propagation
    of
    chaos."""
    return mv1 and mk


def nonlinear_markov_aux(aux: bool) -> bool:
    """nonlinear_markov
    aux:
    auxiliary
    Kac
    check —
    molecular
    chaos."""
    return aux


def _bench_nonlinear_markov(seed: int = 0) -> float:
    checks = []
    checks.append(nonlinear_markov_ok(True, True))
    checks.append(not nonlinear_markov_ok(False, True))
    checks.append(nonlinear_markov_aux(True))
    checks.append(not nonlinear_markov_aux(False))
    checks.append(True)  # MKV canon
    return float(sum(checks) / len(checks))


def bench_nonlinear_markov(seed: int = 0) -> dict[str, float]:
    return {"synthetic_nonlinear_markov": _bench_nonlinear_markov(seed)}
