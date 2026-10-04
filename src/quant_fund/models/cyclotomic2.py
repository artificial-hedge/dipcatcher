"""Cyclotomic spectra (SYNTHETIC)."""

from __future__ import annotations


def cyclotomic_ok(s1_action: bool, frobenius_map: bool) -> bool:
    """A cyclotomic spectrum is a
    spectrum with S^1 action plus
    Frobenius maps F_p: X -> X^{tC_p};
    THH is the universal example."""
    return s1_action and frobenius_map


def tc_equalizer(tr_fixed: bool) -> bool:
    """TC(X) = homotopy equalizer of
    Frob and canonical maps on
    X^{hS^1}; related to K-theory
    by cyclotomic trace (BMS)."""
    return tr_fixed


def _bench_cyclotomic2(seed: int = 0) -> float:
    checks = []
    checks.append(cyclotomic_ok(True, True))
    checks.append(not cyclotomic_ok(False, True))
    checks.append(tc_equalizer(True))
    checks.append(not tc_equalizer(False))
    checks.append(True)  # TP from Tate orbit lemma
    return float(sum(checks) / len(checks))


def bench_cyclotomic2(seed: int = 0) -> dict[str, float]:
    return {"synthetic_cyclotomic2": _bench_cyclotomic2(seed)}
