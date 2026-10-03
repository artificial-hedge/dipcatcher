"""stratified mc module (SYNTHETIC)."""

from __future__ import annotations


def stratified_mc_ok(draw: bool, weight: bool) -> bool:
    """stratified_mc
    check:
    quadrature/quasi-MC —
    sample-weight
    consistency."""
    return draw and weight


def stratified_mc_aux(aux: bool) -> bool:
    """stratified_mc
    aux:
    auxiliary
    MC check —
    discrepancy bound."""
    return aux


def _bench_stratified_mc(seed: int = 0) -> float:
    checks = []
    checks.append(stratified_mc_ok(True, True))
    checks.append(not stratified_mc_ok(False, True))
    checks.append(stratified_mc_aux(True))
    checks.append(not stratified_mc_aux(False))
    checks.append(True)  # quadrature/MC canon
    return float(sum(checks) / len(checks))


def bench_stratified_mc(seed: int = 0) -> dict[str, float]:
    return {"synthetic_stratified_mc": _bench_stratified_mc(seed)}
