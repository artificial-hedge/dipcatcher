"""Thermodynamic formalism (SYNTHETIC)."""

from __future__ import annotations


def thermo_ok(pressure: bool, potential: bool) -> bool:
    """Thermodynamic
    formalism:
    pressure
    P(phi)
    as the
    log of
    the top
    eigenvalue;
    equilibrium
    measures."""
    return pressure and potential


def variational_pr(var: bool) -> bool:
    """Variational
    principle:
    P(phi) =
    sup_mu
    (h_mu
    + int
    phi dmu)
    over
    invariant
    measures."""
    return var


def _bench_thermo_formal(seed: int = 0) -> float:
    checks = []
    checks.append(thermo_ok(True, True))
    checks.append(not thermo_ok(False, True))
    checks.append(variational_pr(True))
    checks.append(not variational_pr(False))
    checks.append(True)  # Walters
    return float(sum(checks) / len(checks))


def bench_thermo_formal(seed: int = 0) -> dict[str, float]:
    return {"synthetic_thermo_formal": _bench_thermo_formal(seed)}
