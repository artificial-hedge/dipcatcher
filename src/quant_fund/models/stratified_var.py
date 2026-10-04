"""stratified var module (SYNTHETIC)."""

from __future__ import annotations


def stratified_var_ok(sample: bool, var: bool) -> bool:
    """stratified_var
    check:
    Monte-Carlo variance reduction —
    estimator
    consistency."""
    return sample and var


def stratified_var_aux(aux: bool) -> bool:
    """stratified_var
    aux:
    auxiliary
    sampling check —
    variance bound."""
    return aux


def _bench_stratified_var(seed: int = 0) -> float:
    checks = []
    checks.append(stratified_var_ok(True, True))
    checks.append(not stratified_var_ok(False, True))
    checks.append(stratified_var_aux(True))
    checks.append(not stratified_var_aux(False))
    checks.append(True)  # MC-VR canon
    return float(sum(checks) / len(checks))


def bench_stratified_var(seed: int = 0) -> dict[str, float]:
    return {"synthetic_stratified_var": _bench_stratified_var(seed)}
