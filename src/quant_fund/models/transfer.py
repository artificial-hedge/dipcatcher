"""Transfer maps for finite covers (SYNTHETIC)."""

from __future__ import annotations


def transfer_push(cover_deg: int, class_wt: int) -> int:
    """p_* tau(a) = deg(p) * a: transfer then pushforward multiplies by
    the covering degree."""
    return cover_deg * class_wt


def transfer_lift(class_wt: int, n_sheets: int) -> int:
    """The transfer of a downstairs class = sum of its n_sheets lifts."""
    return class_wt * n_sheets


def _bench_transfer(seed: int = 0) -> float:
    checks = []
    # double cover S1 -> S1: transfer of the fundamental class = 2 lifts,
    # pushed back gives 2 [S1]
    checks.append(transfer_push(2, 1) == 2)
    checks.append(transfer_lift(1, 2) == 2)
    # 3-fold cover of the circle: push(transfer(1)) = 3
    checks.append(transfer_push(3, 1) == 3)
    # Euler characteristic is multiplicative under covers:
    # chi(S1 double cover of wedge_2 S1) = 2 * chi(wedge2) = 2*(1-2)=-2
    chi_base = 1 - 2
    checks.append(transfer_lift(chi_base, 2) == -2)
    # chi of 3-fold cover of wedge_3 S1 = 3*(1-3) = -6
    checks.append(transfer_lift(1 - 3, 3) == -6)
    # degree-1 cover is the identity
    checks.append(transfer_push(1, 5) == 5)
    return float(sum(checks) / len(checks))


def bench_transfer(seed: int = 0) -> dict[str, float]:
    return {"synthetic_transfer": _bench_transfer(seed)}
