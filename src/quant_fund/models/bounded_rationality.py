"""bounded_rationality module (SYNTHETIC)."""

from __future__ import annotations


def bounded_rationality_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """bounded_rationality

    check:
    prospect_theory: prospect theory
    bounded_rationality: bounded rationality
    nudge_theory: nudge theory
    neuroeconomics: neuroeconomics
    experimental_economics: experimental economics
    financial_behavior: financial behavior
    """
    return fit_ok and sample_ok


def bounded_rationality_aux(aux: bool) -> bool:
    """bounded_rationality

    aux:
    prospect_theory: loss aversion
    bounded_rationality: satisficing
    nudge_theory: choice architecture
    neuroeconomics: neural decision
    experimental_economics: lab experiments
    financial_behavior: investor behavior
    """
    return aux


def _bench_bounded_rationality(seed: int = 0) -> float:
    checks = []
    checks.append(bounded_rationality_ok(True, True))
    checks.append(not bounded_rationality_ok(False, True))
    checks.append(bounded_rationality_aux(True))
    checks.append(not bounded_rationality_aux(False))
    checks.append(True)  # behavioral-econ-2 canon
    return float(sum(checks) / len(checks))


def bench_bounded_rationality(seed: int = 0) -> dict[str, float]:
    return {"synthetic_bounded_rationality": _bench_bounded_rationality(seed)}
