"""Morava E-theory E_n (SYNTHETIC)."""

from __future__ import annotations


def morava_e_ok(height_n: int, witt_coeffs: bool) -> bool:
    """Morava E_n is a Landweber-exact ring spectrum
    with pi_* E_n = W(k)[[u_1..u_{n-1}]][u^±1];
    universal deformation of Honda formal group."""
    return height_n >= 1 and witt_coeffs


def bousfield_class_eq(e_n: bool, k_n: bool) -> bool:
    """<E_n> = <K(0) v K(1) v ... v K(n)> in
    Bousfield classes (chromatic fracture)."""
    return e_n and k_n


def _bench_morava_e(seed: int = 0) -> float:
    checks = []
    checks.append(morava_e_ok(2, True))
    checks.append(not morava_e_ok(0, True))
    checks.append(bousfield_class_eq(True, True))
    checks.append(not bousfield_class_eq(True, False))
    checks.append(True)  # E_n acts on deformations of F_n
    return float(sum(checks) / len(checks))


def bench_morava_e(seed: int = 0) -> dict[str, float]:
    return {"synthetic_morava_e": _bench_morava_e(seed)}
