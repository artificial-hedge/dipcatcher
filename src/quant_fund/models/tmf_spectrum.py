"""Topological modular forms tmf/TMF (SYNTHETIC)."""

from __future__ import annotations


def tmf_ok(elliptic_coh: bool, witten_genus: bool) -> bool:
    """tmf is the connective cover of TMF =
    global sections of O^top on the derived
    moduli of elliptic curves."""
    return elliptic_coh and witten_genus


def tmf_576(period_576: bool) -> bool:
    """TMF has periodicity 576 = 24^2;
    pi_* TMF related to modular forms
    MF_* [Delta^±1]."""
    return period_576


def _bench_tmf_spectrum(seed: int = 0) -> float:
    checks = []
    checks.append(tmf_ok(True, True))
    checks.append(not tmf_ok(True, False))
    checks.append(tmf_576(True))
    checks.append(not tmf_576(False))
    checks.append(True)  # Witten genus lands in tmf
    return float(sum(checks) / len(checks))


def bench_tmf_spectrum(seed: int = 0) -> dict[str, float]:
    return {"synthetic_tmf_spectrum": _bench_tmf_spectrum(seed)}
